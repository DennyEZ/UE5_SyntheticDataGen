"""
UE5.7.1 YOLO Dataset Generator V3 — Per-Object, Registry-Driven

Generates synthetic training data for YOLO object detection or segmentation,
one object class at a time. Each object produces a standalone YOLO-compatible
dataset folder (class 0 = that object). Use merge_datasets.py to combine
multiple per-object datasets into a multi-class training set.

Key features over V2:
- Per-object generation with individual output folders (cam_group/object/)
- Horizontal hemisphere trajectory (for cam_front objects)
- Vertical hemisphere trajectory (for cam_bottom objects)
- Registry-driven config (object_registry.py)
- Supports both detection and segmentation label formats
- In-editor "flag" mechanism via YOLO_V3_GENERATE in config.py

Output structure:
    YOLO_V3_OUTPUT_ROOT/
    ├── cam_front/
    │   ├── gate_sawfish/   (class 0 = gate_sawfish)
    │   │   ├── data.yaml
    │   │   ├── train/images/ & labels/
    │   │   └── val/images/ & labels/
    │   └── ...
    ├── cam_bottom/...
    └── cam_bottom_seg/...

Prerequisites:
    Install opencv in UE5's Python:
        <UE5_ROOT>/Engine/Binaries/ThirdParty/Python3/Win64/python.exe -m pip install opencv-python-headless
"""

import unreal
import math
import os
import shutil
import random
import glob
import json

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# =============================================================================
# CONFIGURATION
# =============================================================================
import sys
import importlib
if '__file__' in dir():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    _script_dir = next((p for p in [os.getcwd()] + sys.path if os.path.isfile(os.path.join(p, 'config.py'))), '')
if _script_dir and _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
# config.py lives in the parent directory (repo root)
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir and _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Force-reload config modules so edits take effect without restarting UE5
import config as _config_mod
importlib.reload(_config_mod)
import object_registry as _registry_mod
importlib.reload(_registry_mod)

from config import (
    TARGET_TAG, CAMERA_TAG, IGNORE_TAG, HIDE_IN_NEGATIVE_TAG,
    POOL_BOUNDS, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM, FOCAL_LENGTH_MM,
    RESOLUTION_X, RESOLUTION_Y, WARMUP_FRAMES, SPATIAL_SAMPLES, TEMPORAL_SAMPLES,
    YOLO_V3_GENERATE, YOLO_V3_OUTPUT_ROOT, YOLO_V3_SEQUENCE_PREFIX, YOLO_V3_MODE,
)
from object_registry import get_object_config, resolve_targets

# Internal constants
# Show-only mask (detect) needs only 1 capture, so full res is affordable.
# Segment mode also uses full resolution for accurate polygon vertices.
MASK_RESOLUTION_X = RESOLUTION_X
MASK_RESOLUTION_Y = RESOLUTION_Y
RT_ASSET_PATH = "/Game/Generated/YOLOV3MaskRT"
SEQUENCE_PATH = YOLO_V3_SEQUENCE_PREFIX + "Sequence"

# Global reference to prevent garbage collection
global_executor = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_world():
    try:
        subsys = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        return subsys.get_editor_world()
    except (AttributeError, Exception):
        return unreal.EditorLevelLibrary.get_editor_world()


def calculate_intrinsics():
    px_per_mm_x = RESOLUTION_X / SENSOR_WIDTH_MM
    px_per_mm_y = RESOLUTION_Y / SENSOR_HEIGHT_MM
    return {
        "fx": FOCAL_LENGTH_MM * px_per_mm_x,
        "fy": FOCAL_LENGTH_MM * px_per_mm_y,
        "cx": RESOLUTION_X / 2.0,
        "cy": RESOLUTION_Y / 2.0,
    }


def project_point(world_pt, cam_transform, intrinsics):
    cam_inv = cam_transform.inverse()
    local = unreal.MathLibrary.transform_location(cam_inv, world_pt)
    cv_x, cv_y, cv_z = local.y, -local.z, local.x
    if cv_z <= 0:
        return [-9999.0, -9999.0]
    u = (cv_x * intrinsics["fx"] / cv_z) + intrinsics["cx"]
    v = (cv_y * intrinsics["fy"] / cv_z) + intrinsics["cy"]
    return [u, v]


def _clamp_to_bounds(pos):
    return unreal.Vector(
        max(POOL_BOUNDS["x_min"], min(pos.x, POOL_BOUNDS["x_max"])),
        max(POOL_BOUNDS["y_min"], min(pos.y, POOL_BOUNDS["y_max"])),
        max(POOL_BOUNDS["z_min"], min(pos.z, POOL_BOUNDS["z_max"]))
    )


# =============================================================================
# CAMERA POSITION GENERATORS
# =============================================================================

def generate_vertical_hemisphere(center, min_dist, max_dist):
    """Camera orbits ABOVE — bird's eye to side views (for cam_bottom)."""
    dist = random.uniform(min_dist, max_dist)
    theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(random.uniform(0, 1))  # uniform hemisphere
    dx = dist * math.sin(phi) * math.cos(theta)
    dy = dist * math.sin(phi) * math.sin(theta)
    dz = dist * math.cos(phi)
    return _clamp_to_bounds(unreal.Vector(center.x + dx, center.y + dy, center.z + dz))


def generate_horizontal_hemisphere(center, min_dist, max_dist, theta_range=None):
    """Camera orbits AROUND at eye level (for cam_front).

    phi range ~60°–90° from vertical axis. 20% chance of looking up.
    theta_range: optional (min_deg, max_deg) to restrict azimuthal angle.
    """
    dist = random.uniform(min_dist, max_dist)
    if theta_range is not None:
        theta = math.radians(random.uniform(theta_range[0], theta_range[1]))
    else:
        theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(random.uniform(0, 0.5))  # bias toward horizontal
    dx = dist * math.sin(phi) * math.cos(theta)
    dy = dist * math.sin(phi) * math.sin(theta)
    dz = dist * math.cos(phi)
    if random.random() < 0.2:
        dz = -dz  # occasionally look up at object
    return _clamp_to_bounds(unreal.Vector(center.x + dx, center.y + dy, center.z + dz))


def generate_camera_position(center, obj_config):
    min_d, max_d = obj_config["min_distance"], obj_config["max_distance"]
    if obj_config["hemisphere"] == "horizontal":
        return generate_horizontal_hemisphere(center, min_d, max_d,
                                              theta_range=obj_config.get("theta_range"))
    return generate_vertical_hemisphere(center, min_d, max_d)


# =============================================================================
# SCENE CAPTURE MASK PIPELINE
# =============================================================================

def create_render_target_asset(width, height):
    if unreal.EditorAssetLibrary.does_asset_exist(RT_ASSET_PATH):
        unreal.EditorAssetLibrary.delete_asset(RT_ASSET_PATH)
    world = get_world()
    try:
        rt = unreal.RenderingLibrary.create_render_target2d(
            world, width=width, height=height,
            format=unreal.TextureRenderTargetFormat.RTF_RGBA8,
            clear_color=unreal.LinearColor(0, 0, 0, 1))
        if rt:
            return rt
    except (AttributeError, Exception):
        pass
    try:
        pkg_path, asset_name = RT_ASSET_PATH.rsplit('/', 1)
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
        factory = unreal.CanvasRenderTarget2DFactoryNew()
        rt = asset_tools.create_asset(asset_name, pkg_path, unreal.CanvasRenderTarget2D, factory)
        if rt:
            rt.set_editor_property('size_x', width)
            rt.set_editor_property('size_y', height)
            rt.set_editor_property('render_target_format', unreal.TextureRenderTargetFormat.RTF_RGBA8)
            rt.set_editor_property('clear_color', unreal.LinearColor(0, 0, 0, 1))
            return rt
    except (AttributeError, Exception):
        pass
    unreal.log_error("Failed to create render target!")
    return None


def setup_scene_capture(camera_actor):
    """Set up SceneCapture2D for segment mode (differential masking)."""
    rt = create_render_target_asset(MASK_RESOLUTION_X, MASK_RESOLUTION_Y)
    if not rt:
        return None, None
    try:
        fov_degrees = camera_actor.get_cine_camera_component().field_of_view
    except (AttributeError, Exception):
        fov_degrees = math.degrees(2.0 * math.atan(SENSOR_WIDTH_MM / (2.0 * FOCAL_LENGTH_MM)))
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    capture_actor = subsys.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(), unreal.Rotator())
    if not capture_actor:
        return None, None
    cc = capture_actor.capture_component2d
    cc.texture_target = rt
    cc.set_editor_property('primitive_render_mode',
                           unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES)
    cc.set_editor_property('capture_source', unreal.SceneCaptureSource.SCS_BASE_COLOR)
    cc.set_editor_property('fov_angle', fov_degrees)
    cc.set_editor_property('capture_every_frame', False)
    cc.set_editor_property('capture_on_movement', False)
    return capture_actor, rt


def _read_render_target_as_numpy(render_target):
    world = get_world()
    colors = unreal.RenderingLibrary.read_render_target(world, render_target, normalize=True)
    if colors is None:
        return None
    w, h = render_target.size_x, render_target.size_y
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, c in enumerate(colors):
        pixels[idx // w, idx % w] = [c.b, c.g, c.r]
    return pixels


def capture_differential_mask(capture_actor, render_target, cine_camera,
                               cam_pos, cam_rot, target_actor):
    """Two-pass differential mask. Returns binary mask (HxW uint8) or None."""
    cc = capture_actor.capture_component2d
    cine_camera.set_actor_location_and_rotation(cam_pos, cam_rot, False, True)
    try:
        cine_comp = cine_camera.get_cine_camera_component()
        actual_pos = cine_comp.get_world_location()
        actual_rot = cine_comp.get_world_rotation()
    except (AttributeError, Exception):
        actual_pos, actual_rot = cam_pos, cam_rot
    cc.set_world_location_and_rotation(actual_pos, actual_rot, False, True)

    # Pass 1: background (target hidden)
    target_actor.set_actor_hidden_in_game(True)
    cc.capture_scene()
    bg = _read_render_target_as_numpy(render_target)

    # Pass 2: foreground (target visible)
    target_actor.set_actor_hidden_in_game(False)
    cc.capture_scene()
    fg = _read_render_target_as_numpy(render_target)

    if bg is None or fg is None:
        return None
    diff = cv2.absdiff(fg, bg)
    diff_gray = np.max(diff, axis=2)
    _, binary = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
    # Open removes small noise spots without expanding the mask outward
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def extract_polygons_from_mask(binary_mask, epsilon_factor=0.002, min_area=6):
    """Returns list of [(x,y)...] normalized polygon points, or None."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    polygons = []
    h, w = binary_mask.shape[:2]
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        approx = cv2.approxPolyDP(contour, epsilon_factor * cv2.arcLength(contour, True), True)
        if len(approx) < 3:
            continue
        poly = [(max(0.0, min(1.0, float(p[0][0]) / w)),
                 max(0.0, min(1.0, float(p[0][1]) / h))) for p in approx]
        polygons.append(poly)
    return polygons if polygons else None


def _get_annotation_world_corners(actor):
    """Return world-space corners for labeling, preferring a tagged proxy box.

    If the actor contains a `BoxComponent` tagged `DOPE_Bounds`, its transform
    and extents are used as the annotation volume. Otherwise, all visible
    `StaticMeshComponent`s are unioned. Final fallback is actor AABB.
    """
    for comp in actor.get_components_by_class(unreal.BoxComponent):
        if comp.component_has_tag("DOPE_Bounds"):
            extent = comp.get_unscaled_box_extent() * comp.get_world_scale()
            ex, ey, ez = extent.x, extent.y, extent.z
            local_corners = [
                unreal.Vector(sx * ex, sy * ey, sz * ez)
                for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)
            ]
            comp_tf = comp.get_world_transform()
            return [unreal.MathLibrary.transform_location(comp_tf, c) for c in local_corners]

    all_corners = []
    for mesh_comp in actor.get_components_by_class(unreal.StaticMeshComponent):
        if not mesh_comp.static_mesh:
            continue
        if not mesh_comp.is_visible():
            continue
        mesh_bounds = mesh_comp.static_mesh.get_bounds()
        lo = mesh_bounds.origin
        le = mesh_bounds.box_extent
        ex, ey, ez = le.x, le.y, le.z
        ox, oy, oz = lo.x, lo.y, lo.z
        local_corners = [
            unreal.Vector(ox + sx * ex, oy + sy * ey, oz + sz * ez)
            for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)
        ]
        comp_tf = mesh_comp.get_world_transform()
        all_corners.extend(
            unreal.MathLibrary.transform_location(comp_tf, c) for c in local_corners
        )

    if all_corners:
        return all_corners

    origin, extent = actor.get_actor_bounds(False)
    ex, ey, ez = extent.x, extent.y, extent.z
    ox, oy, oz = origin.x, origin.y, origin.z
    return [
        unreal.Vector(ox + sx * ex, oy + sy * ey, oz + sz * ez)
        for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)
    ]


def _describe_annotation_source(actor):
    """Return a short description of which bounds source labeling will use."""
    for comp in actor.get_components_by_class(unreal.BoxComponent):
        if comp.component_has_tag("DOPE_Bounds"):
            extent = comp.get_unscaled_box_extent() * comp.get_world_scale()
            return (
                f"DOPE_Bounds BoxComponent '{comp.get_name()}' "
                f"extent=({extent.x:.1f}, {extent.y:.1f}, {extent.z:.1f})"
            )

    mesh_comps = [
        comp for comp in actor.get_components_by_class(unreal.StaticMeshComponent)
        if comp.static_mesh and comp.is_visible()
    ]
    if mesh_comps:
        if len(mesh_comps) == 1:
            mesh_name = mesh_comps[0].static_mesh.get_name()
            return f"StaticMeshComponent mesh bounds '{mesh_name}'"
        return f"union of {len(mesh_comps)} visible StaticMeshComponents"

    return "actor AABB fallback"


def _get_annotation_center(actor):
    corners = _get_annotation_world_corners(actor)
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return unreal.Vector(
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
        (min(zs) + max(zs)) / 2.0,
    )


def _get_2d_bbox_obb(actor, cam_transform, intrinsics):
    """Tight 2D bounding box via OBB projection of ALL mesh components.

    Iterates every StaticMeshComponent on the actor, transforms each mesh's
    local bounding box corners to world space, projects to 2D, and takes the
    union.  This handles Blueprint actors with multiple sub-meshes, child
    components, and avoids being thrown off by a single small part.
    """
    all_corners = _get_annotation_world_corners(actor)

    pts = [p for p in (project_point(c, cam_transform, intrinsics) for c in all_corners)
           if p != [-9999.0, -9999.0]]
    if len(pts) < 4:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2 = max(0, min(xs)), min(RESOLUTION_X, max(xs))
    y1, y2 = max(0, min(ys)), min(RESOLUTION_Y, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        max(0.0, min(1.0, ((x1 + x2) / 2.0) / RESOLUTION_X)),
        max(0.0, min(1.0, ((y1 + y2) / 2.0) / RESOLUTION_Y)),
        max(0.0, min(1.0, (x2 - x1) / RESOLUTION_X)),
        max(0.0, min(1.0, (y2 - y1) / RESOLUTION_Y)),
    )


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class YOLOv3DatasetGenerator:
    def __init__(self):
        self.camera = None
        self.all_target_actors = []
        self.negative_hide_actors = []
        self.intrinsics = calculate_intrinsics()
        self.object_queue = []
        self.objects_completed = []

        # Per-object state (reset each iteration)
        self.current_target = None
        self.current_config = None
        self.current_output_dir = ""
        self.current_sample_data = []
        self.current_total_samples = 0
        self.staging_images = ""
        self.staging_labels = ""
        self.non_target_original_locs = {}
        self.current_co_visible = []      # list of (canonical_name, actor) tuples
        self.current_class_map = {}       # {canonical_name: class_id}

        unreal.log("=" * 60)
        unreal.log("UE5.7 YOLO DATASET GENERATOR V3")
        unreal.log("  Per-Object | Registry-Driven | Multi-Hemisphere")
        unreal.log("=" * 60)

        if YOLO_V3_MODE == "segment" and not HAS_CV2:
            unreal.log_error("Segmentation mode requires cv2!")
            return

        self._find_actors()
        if not self.camera:
            unreal.log_error(f"No camera with tag '{CAMERA_TAG}' found!")
            return

        self._snapshot_initial_transforms()
        self._configure_camera()

        try:
            self.object_queue = resolve_targets(YOLO_V3_GENERATE)
        except ValueError as e:
            unreal.log_error(f"Invalid GENERATE config: {e}")
            return

        unreal.log(f"Mode: {YOLO_V3_MODE}")
        unreal.log(f"Objects ({len(self.object_queue)}): {self.object_queue}")
        unreal.log(f"Output: {YOLO_V3_OUTPUT_ROOT}")
        unreal.log("")

        self._process_next_object()

    # -------------------------------------------------------------------------
    # Actor Discovery
    # -------------------------------------------------------------------------

    def _find_actors(self):
        subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        stale = 0
        ignored = 0
        negative_hide = 0
        for actor in subsys.get_all_level_actors():
            if actor.actor_has_tag(TARGET_TAG):
                self.all_target_actors.append(actor)
            elif actor.actor_has_tag(CAMERA_TAG):
                self.camera = actor
            elif actor.actor_has_tag(IGNORE_TAG):
                actor.destroy_actor()
                ignored += 1
            elif actor.actor_has_tag(HIDE_IN_NEGATIVE_TAG):
                self.negative_hide_actors.append(actor)
                negative_hide += 1
            elif actor.get_class().get_name() == "SceneCapture2D":
                actor.destroy_actor()
                stale += 1
        if stale:
            unreal.log(f"  Cleaned {stale} stale SceneCapture2D(s)")
        if ignored:
            unreal.log(f"  Removed {ignored} ignored object(s)")
        if negative_hide:
            unreal.log(f"  Found {negative_hide} negative-hide actor(s)")
        unreal.log(f"  Found {len(self.all_target_actors)} target(s), camera: {'Yes' if self.camera else 'No'}")

    def _configure_camera(self):
        if not self.camera:
            return
        cc = self.camera.get_cine_camera_component()
        cc.filmback.sensor_width = SENSOR_WIDTH_MM
        cc.filmback.sensor_height = SENSOR_HEIGHT_MM
        cc.current_focal_length = FOCAL_LENGTH_MM
        cc.focus_settings.focus_method = unreal.CameraFocusMethod.DISABLE

    def _snapshot_initial_transforms(self):
        """Save every tracked actor's transform so we can restore after generation."""
        self.initial_transforms = {}
        for actor in self.all_target_actors + self.negative_hide_actors:
            self.initial_transforms[actor] = (
                actor.get_actor_location(),
                actor.get_actor_rotation(),
            )
        # Persist to disk for crash recovery
        self._save_transforms_to_disk()
        unreal.log(f"  Saved initial transforms for {len(self.initial_transforms)} actor(s)")

    def _save_transforms_to_disk(self):
        """Write initial transforms to JSON so restore_scene() can recover from crashes."""
        data = {}
        for actor, (loc, rot) in self.initial_transforms.items():
            label = actor.get_actor_label()
            data[label] = {
                "x": loc.x, "y": loc.y, "z": loc.z,
                "roll": rot.roll, "pitch": rot.pitch, "yaw": rot.yaw,
            }
        os.makedirs(YOLO_V3_OUTPUT_ROOT, exist_ok=True)
        path = os.path.join(YOLO_V3_OUTPUT_ROOT, "_initial_transforms.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _restore_all_initial_transforms(self):
        """Restore every tracked actor to its position at script start."""
        count = 0
        for actor, (loc, rot) in self.initial_transforms.items():
            try:
                actor.set_actor_location_and_rotation(loc, rot, False, True)
                count += 1
            except Exception:
                pass  # actor may have been destroyed
        # Clean up the crash-recovery file
        path = os.path.join(YOLO_V3_OUTPUT_ROOT, "_initial_transforms.json")
        if os.path.exists(path):
            os.remove(path)
        unreal.log(f"  Restored {count} actor(s) to initial transforms")

    def _find_actor_by_label(self, label):
        for actor in self.all_target_actors:
            if actor.get_actor_label() == label:
                return actor
        return None

    # -------------------------------------------------------------------------
    # Per-Object Processing Loop
    # -------------------------------------------------------------------------

    def _process_next_object(self):
        if not self.object_queue:
            self._on_all_complete()
            return

        obj_name = self.object_queue.pop(0)
        obj_config = get_object_config(obj_name)

        unreal.log("=" * 60)
        unreal.log(f"GENERATING: {obj_name}")
        unreal.log(f"  Group: {obj_config['camera_group']} | "
                   f"Hemisphere: {obj_config['hemisphere']} | "
                   f"Samples: {obj_config['samples']} | "
                   f"Dist: {obj_config['min_distance']}–{obj_config['max_distance']}cm | "
                   f"Placement: {'randomized' if obj_config.get('placement') else 'fixed'}")
        unreal.log("=" * 60)

        target = self._find_actor_by_label(obj_config["actor_label"])
        if not target:
            unreal.log_warning(
                f"  Actor '{obj_config['actor_label']}' not found — skipping '{obj_name}'.\n"
                f"  Available: {[a.get_actor_label() for a in self.all_target_actors]}")
            self._process_next_object()
            return

        unreal.log(f"  Annotation bounds: {_describe_annotation_source(target)}")

        self.current_target = target
        self.current_config = obj_config
        self.current_sample_data = []

        # Resolve co_visible actors
        self.current_co_visible = []
        for co_name in obj_config.get("co_visible", []):
            try:
                co_cfg = get_object_config(co_name)
                co_actor = self._find_actor_by_label(co_cfg["actor_label"])
                if co_actor:
                    self.current_co_visible.append((co_name, co_actor))
                else:
                    unreal.log_warning(f"  Co-visible actor '{co_name}' not found in scene")
            except KeyError:
                unreal.log_warning(f"  Co-visible '{co_name}' not in registry")

        # Build class map: target=0, co_visible sorted alphabetically=1,2,...
        self.current_class_map = {obj_name: 0}
        for i, (co_name, _) in enumerate(sorted(self.current_co_visible)):
            self.current_class_map[co_name] = i + 1
        if self.current_co_visible:
            unreal.log(f"  Co-visible: {[n for n, _ in self.current_co_visible]} → class map: {self.current_class_map}")

        # Resolve keep_visible labels: collect from target + co-visible configs
        self.current_keep_visible_labels = set(obj_config.get("keep_visible", []))
        for co_name, _ in self.current_co_visible:
            try:
                co_cfg = get_object_config(co_name)
                self.current_keep_visible_labels.update(co_cfg.get("keep_visible", []))
            except KeyError:
                pass
        if self.current_keep_visible_labels:
            unreal.log(f"  Keep-visible HideInNegative actors: {sorted(self.current_keep_visible_labels)}")

        cam_group = obj_config["camera_group"]
        self.current_output_dir = os.path.join(YOLO_V3_OUTPUT_ROOT, cam_group, obj_name)
        if os.path.exists(self.current_output_dir):
            shutil.rmtree(self.current_output_dir)
        self.staging_images = os.path.join(self.current_output_dir, "_staging", "images")
        self.staging_labels = os.path.join(self.current_output_dir, "_staging", "labels")
        os.makedirs(self.staging_images)
        os.makedirs(self.staging_labels)

        num_positive = obj_config["samples"]
        neg_ratio = obj_config.get("negative_ratio", 0.1)
        num_negative = int(num_positive * neg_ratio / max(1 - neg_ratio, 0.01))
        self.current_total_samples = num_positive + num_negative
        unreal.log(f"  {num_positive} pos + {num_negative} neg = {self.current_total_samples} total")

        self._hide_non_targets(target)

        sequence = self._create_sequence(target, obj_config, obj_name)
        if not sequence:
            self._restore_non_targets()
            self._process_next_object()
            return

        self._generate_labels(target, obj_config, obj_name)
        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
        self._render(sequence, obj_name, obj_config)

    def _hide_non_targets(self, target_actor):
        co_visible_actors = {a for _, a in self.current_co_visible}
        self.non_target_original_locs = {}
        for actor in self.all_target_actors:
            if actor != target_actor and actor not in co_visible_actors:
                self.non_target_original_locs[actor] = actor.get_actor_location()
                loc = actor.get_actor_location()
                actor.set_actor_location(unreal.Vector(loc.x, loc.y, -20000.0), False, False)
        # Also hide HideInNegative actors not listed in keep_visible
        hidden_hide_count = 0
        for actor in self.negative_hide_actors:
            if actor.get_actor_label() not in self.current_keep_visible_labels:
                self.non_target_original_locs[actor] = actor.get_actor_location()
                loc = actor.get_actor_location()
                actor.set_actor_location(unreal.Vector(loc.x, loc.y, -20000.0), False, False)
                hidden_hide_count += 1
        if self.non_target_original_locs:
            unreal.log(f"  Hidden {len(self.non_target_original_locs)} non-target actor(s)"
                       f" ({hidden_hide_count} HideInNegative)")
        if co_visible_actors:
            unreal.log(f"  Kept {len(co_visible_actors)} co-visible actor(s) above ground")

    def _restore_non_targets(self):
        for actor, orig_loc in self.non_target_original_locs.items():
            actor.set_actor_location(orig_loc, False, False)
        count = len(self.non_target_original_locs)
        self.non_target_original_locs = {}
        if count:
            unreal.log(f"  Restored {count} non-target actor(s)")

    # -------------------------------------------------------------------------
    # Sequence Creation
    # -------------------------------------------------------------------------

    def _create_sequence(self, target, obj_config, obj_name):
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
        if unreal.EditorAssetLibrary.does_asset_exist(SEQUENCE_PATH):
            unreal.EditorAssetLibrary.delete_asset(SEQUENCE_PATH)

        pkg_path, asset_name = SEQUENCE_PATH.rsplit('/', 1)
        seq = asset_tools.create_asset(
            asset_name, pkg_path, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
        seq.set_display_rate(unreal.FrameRate(24, 1))

        # Bind camera
        cam_binding = seq.add_possessable(self.camera)
        cam_track = cam_binding.add_track(unreal.MovieScene3DTransformTrack)
        cam_section = cam_track.add_section()

        frames_per_sample = 1
        total_frames = self.current_total_samples * frames_per_sample
        cam_section.set_range(0, total_frames + 10)
        seq.set_playback_start(0)
        seq.set_playback_end(total_frames)
        cam_channels = cam_section.get_all_channels()

        # Bind target actor
        target_binding = seq.add_possessable(target)
        target_track = target_binding.add_track(unreal.MovieScene3DTransformTrack)
        target_section = target_track.add_section()
        target_section.set_range(0, total_frames + 10)
        target_channels = target_section.get_all_channels()

        orig_loc = target.get_actor_location()
        orig_rot = target.get_actor_rotation()

        # Initial keyframe at frame 0
        frame_0 = unreal.FrameNumber(0)
        target_channels[0].add_key(frame_0, orig_loc.x)
        target_channels[1].add_key(frame_0, orig_loc.y)
        target_channels[2].add_key(frame_0, orig_loc.z)
        target_channels[3].add_key(frame_0, orig_rot.roll)
        target_channels[4].add_key(frame_0, orig_rot.pitch)
        target_channels[5].add_key(frame_0, orig_rot.yaw)

        # Bind co-visible actors (keyframed: original pos for positive, underground for negative)
        co_visible_tracks = []
        for co_name, co_actor in self.current_co_visible:
            co_binding = seq.add_possessable(co_actor)
            co_track = co_binding.add_track(unreal.MovieScene3DTransformTrack)
            co_section = co_track.add_section()
            co_section.set_range(0, total_frames + 10)
            co_channels = co_section.get_all_channels()
            co_loc = co_actor.get_actor_location()
            co_rot = co_actor.get_actor_rotation()
            # Initial keyframe
            co_channels[0].add_key(frame_0, co_loc.x)
            co_channels[1].add_key(frame_0, co_loc.y)
            co_channels[2].add_key(frame_0, co_loc.z)
            co_channels[3].add_key(frame_0, co_rot.roll)
            co_channels[4].add_key(frame_0, co_rot.pitch)
            co_channels[5].add_key(frame_0, co_rot.yaw)
            co_visible_tracks.append({
                'channels': co_channels,
                'orig_loc': co_loc,
                'orig_rot': co_rot,
            })

        # Bind non-target background actors that should disappear only in negatives
        # Only include HideInNegative actors listed in keep_visible for this target
        negative_hide_tracks = []
        for hide_actor in self.negative_hide_actors:
            if hide_actor.get_actor_label() not in self.current_keep_visible_labels:
                continue
            hide_binding = seq.add_possessable(hide_actor)
            hide_track = hide_binding.add_track(unreal.MovieScene3DTransformTrack)
            hide_section = hide_track.add_section()
            hide_section.set_range(0, total_frames + 10)
            hide_channels = hide_section.get_all_channels()
            hide_loc = hide_actor.get_actor_location()
            hide_rot = hide_actor.get_actor_rotation()
            hide_channels[0].add_key(frame_0, hide_loc.x)
            hide_channels[1].add_key(frame_0, hide_loc.y)
            hide_channels[2].add_key(frame_0, hide_loc.z)
            hide_channels[3].add_key(frame_0, hide_rot.roll)
            hide_channels[4].add_key(frame_0, hide_rot.pitch)
            hide_channels[5].add_key(frame_0, hide_rot.yaw)
            negative_hide_tracks.append({
                'actor': hide_actor,
                'channels': hide_channels,
                'orig_loc': hide_loc,
                'orig_rot': hide_rot,
            })
        if negative_hide_tracks:
            unreal.log(f"  Negative-hide actors: {len(negative_hide_tracks)}")

        # Build frame schedule
        num_positive = obj_config["samples"]
        num_negative = self.current_total_samples - num_positive
        frame_types = ['positive'] * num_positive + ['negative'] * num_negative
        random.shuffle(frame_types)

        placement = obj_config.get("placement")
        jitter_enabled = obj_config.get("enable_jitter", True)
        jitter_pitch = obj_config.get("jitter_max_pitch", 5.0)

        for i in range(self.current_total_samples):
            frame_num = i * frames_per_sample
            frame_time = unreal.FrameNumber(frame_num)
            is_negative = (frame_types[i] == 'negative')

            if is_negative:
                # Random camera, target underground
                cam_pos = unreal.Vector(
                    random.uniform(POOL_BOUNDS["x_min"], POOL_BOUNDS["x_max"]),
                    random.uniform(POOL_BOUNDS["y_min"], POOL_BOUNDS["y_max"]),
                    random.uniform(POOL_BOUNDS["z_min"], POOL_BOUNDS["z_max"]))
                cam_rot = unreal.Rotator(
                    roll=0.0,
                    pitch=random.uniform(-70.0, 0.0),
                    yaw=random.uniform(0.0, 360.0))

                target_channels[0].add_key(frame_time, orig_loc.x)
                target_channels[1].add_key(frame_time, orig_loc.y)
                target_channels[2].add_key(frame_time, -20000.0)
                target_channels[3].add_key(frame_time, orig_rot.roll)
                target_channels[4].add_key(frame_time, orig_rot.pitch)
                target_channels[5].add_key(frame_time, orig_rot.yaw)

                # Move co-visible actors underground for negative frames too
                for co_data in co_visible_tracks:
                    co_ch = co_data['channels']
                    co_ch[0].add_key(frame_time, co_data['orig_loc'].x)
                    co_ch[1].add_key(frame_time, co_data['orig_loc'].y)
                    co_ch[2].add_key(frame_time, -20000.0)
                    co_ch[3].add_key(frame_time, co_data['orig_rot'].roll)
                    co_ch[4].add_key(frame_time, co_data['orig_rot'].pitch)
                    co_ch[5].add_key(frame_time, co_data['orig_rot'].yaw)

                for hide_data in negative_hide_tracks:
                    hide_ch = hide_data['channels']
                    hide_ch[0].add_key(frame_time, hide_data['orig_loc'].x)
                    hide_ch[1].add_key(frame_time, hide_data['orig_loc'].y)
                    hide_ch[2].add_key(frame_time, -20000.0)
                    hide_ch[3].add_key(frame_time, hide_data['orig_rot'].roll)
                    hide_ch[4].add_key(frame_time, hide_data['orig_rot'].pitch)
                    hide_ch[5].add_key(frame_time, hide_data['orig_rot'].yaw)

                self.current_sample_data.append({
                    "frame_idx": i, "target": None,
                    "cam_pos": cam_pos, "cam_rot": cam_rot, "is_negative": True})
            else:
                # Compute target position (randomized or original)
                if placement:
                    target_loc = unreal.Vector(
                        orig_loc.x + random.uniform(-placement["xy_range_x"], placement["xy_range_x"]),
                        orig_loc.y + random.uniform(-placement["xy_range_y"], placement["xy_range_y"]),
                        orig_loc.z)
                    target_rot = unreal.Rotator(
                        roll=random.uniform(-placement["roll_range"], placement["roll_range"]),
                        pitch=random.uniform(placement["pitch_min"], placement["pitch_max"]),
                        yaw=random.uniform(0, placement["yaw_range"]))
                else:
                    target_loc = orig_loc
                    target_rot = orig_rot

                target_channels[0].add_key(frame_time, target_loc.x)
                target_channels[1].add_key(frame_time, target_loc.y)
                target_channels[2].add_key(frame_time, target_loc.z)
                target_channels[3].add_key(frame_time, target_rot.roll)
                target_channels[4].add_key(frame_time, target_rot.pitch)
                target_channels[5].add_key(frame_time, target_rot.yaw)

                # Camera aimed at target's bounding box center
                bbox_center = _get_annotation_center(target)
                if placement:
                    offset = unreal.Vector(
                        target_loc.x - orig_loc.x,
                        target_loc.y - orig_loc.y,
                        target_loc.z - orig_loc.z)
                    bbox_center = unreal.Vector(
                        bbox_center.x + offset.x,
                        bbox_center.y + offset.y,
                        bbox_center.z + offset.z)

                cam_pos = generate_camera_position(target_loc, obj_config)
                cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, bbox_center)

                # Camera jitter
                if jitter_enabled:
                    dist = math.sqrt((cam_pos.x - bbox_center.x)**2 +
                                     (cam_pos.y - bbox_center.y)**2 +
                                     (cam_pos.z - bbox_center.z)**2)
                    max_offset = dist * math.tan(math.radians(jitter_pitch))
                    margin = 0.10
                    jitter_scale = 1.0
                    for _ in range(4):
                        off = max_offset * jitter_scale
                        look_pt = unreal.Vector(
                            bbox_center.x + random.uniform(-off, off),
                            bbox_center.y + random.uniform(-off, off),
                            bbox_center.z + random.uniform(-off * 0.5, off * 0.5))
                        test_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, look_pt)
                        test_tf = unreal.Transform(location=cam_pos, rotation=test_rot)
                        c2d = project_point(bbox_center, test_tf, self.intrinsics)
                        if (c2d != [-9999.0, -9999.0] and
                            RESOLUTION_X * margin < c2d[0] < RESOLUTION_X * (1 - margin) and
                            RESOLUTION_Y * margin < c2d[1] < RESOLUTION_Y * (1 - margin)):
                            cam_rot = test_rot
                            break
                        jitter_scale *= 0.5

                # Keyframe co-visible actors at original position for positive frames
                for co_data in co_visible_tracks:
                    co_ch = co_data['channels']
                    co_ch[0].add_key(frame_time, co_data['orig_loc'].x)
                    co_ch[1].add_key(frame_time, co_data['orig_loc'].y)
                    co_ch[2].add_key(frame_time, co_data['orig_loc'].z)
                    co_ch[3].add_key(frame_time, co_data['orig_rot'].roll)
                    co_ch[4].add_key(frame_time, co_data['orig_rot'].pitch)
                    co_ch[5].add_key(frame_time, co_data['orig_rot'].yaw)

                for hide_data in negative_hide_tracks:
                    hide_ch = hide_data['channels']
                    hide_ch[0].add_key(frame_time, hide_data['orig_loc'].x)
                    hide_ch[1].add_key(frame_time, hide_data['orig_loc'].y)
                    hide_ch[2].add_key(frame_time, hide_data['orig_loc'].z)
                    hide_ch[3].add_key(frame_time, hide_data['orig_rot'].roll)
                    hide_ch[4].add_key(frame_time, hide_data['orig_rot'].pitch)
                    hide_ch[5].add_key(frame_time, hide_data['orig_rot'].yaw)

                self.current_sample_data.append({
                    "frame_idx": i, "target": target,
                    "cam_pos": cam_pos, "cam_rot": cam_rot, "is_negative": False})

            # Camera keyframes
            cam_channels[0].add_key(frame_time, cam_pos.x)
            cam_channels[1].add_key(frame_time, cam_pos.y)
            cam_channels[2].add_key(frame_time, cam_pos.z)
            cam_channels[3].add_key(frame_time, cam_rot.roll)
            cam_channels[4].add_key(frame_time, cam_rot.pitch)
            cam_channels[5].add_key(frame_time, cam_rot.yaw)

        # Set constant interpolation on all keyframes
        for ch in cam_channels:
            for key in ch.get_keys():
                key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)
        for ch in target_channels:
            for key in ch.get_keys():
                key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)
        for co_data in co_visible_tracks:
            for ch in co_data['channels']:
                for key in ch.get_keys():
                    key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)
        for hide_data in negative_hide_tracks:
            for ch in hide_data['channels']:
                for key in ch.get_keys():
                    key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)

        # Camera cut track
        camera_cut = self._add_camera_cut_track(seq)
        if camera_cut:
            bid = unreal.MovieSceneObjectBindingID()
            bid.set_editor_property("guid", cam_binding.get_id())
            cs = camera_cut.add_section()
            cs.set_range(0, total_frames + 10)
            cs.set_camera_binding_id(bid)

        unreal.log(f"  Sequence created: {self.current_total_samples} samples, {total_frames} frames")
        return seq

    def _add_camera_cut_track(self, seq):
        movie_scene = seq.get_movie_scene()
        try:
            return movie_scene.add_camera_cut_track()
        except Exception:
            try:
                return seq.add_track(unreal.MovieSceneCameraCutTrack)
            except Exception:
                return None

    # -------------------------------------------------------------------------
    # Label Generation
    # -------------------------------------------------------------------------

    def _generate_labels(self, target, obj_config, obj_name):
        if YOLO_V3_MODE == "segment":
            self._generate_labels_segment(target, obj_config, obj_name)
        else:
            self._generate_labels_detect(target, obj_config, obj_name)

    def _generate_labels_detect(self, target, obj_config, obj_name):
        """Detect mode: pure geometric projection (no SceneCapture needed).

        Projects 3D bounding box corners to 2D via camera intrinsics.
        Fast — no GPU captures, no pixel reads, no cv2 dependency.
        """
        unreal.log(f"  Generating labels (geometric projection, detect)...")

        total_annotations = 0
        empty_frames = 0

        with unreal.ScopedSlowTask(self.current_total_samples,
                                    f"Labels: {obj_name}") as slow_task:
            slow_task.make_dialog(True)
            for data in self.current_sample_data:
                if slow_task.should_cancel():
                    break
                slow_task.enter_progress_frame(1)
                i = data["frame_idx"]
                label_path = os.path.join(self.staging_labels, f"{i:06d}.txt")

                if data["is_negative"]:
                    with open(label_path, 'w') as f:
                        f.write("")
                    empty_frames += 1
                    continue

                cam_pos, cam_rot = data["cam_pos"], data["cam_rot"]
                self.camera.set_actor_location_and_rotation(cam_pos, cam_rot, False, True)
                try:
                    cine_comp = self.camera.get_cine_camera_component()
                    actual_pos = cine_comp.get_world_location()
                    actual_rot = cine_comp.get_world_rotation()
                except (AttributeError, Exception):
                    actual_pos, actual_rot = cam_pos, cam_rot
                cam_tf = unreal.Transform(location=actual_pos, rotation=actual_rot)
                label_lines = []

                actors_to_label = [(self.current_class_map[obj_name], target, obj_name)]
                for co_name, co_actor in self.current_co_visible:
                    actors_to_label.append((self.current_class_map[co_name], co_actor, co_name))

                for class_id, label_actor, label_name in actors_to_label:
                    bbox = _get_2d_bbox_obb(label_actor, cam_tf, self.intrinsics)
                    if bbox:
                        xc, yc, w, h = bbox
                        label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                        total_annotations += 1

                with open(label_path, 'w') as f:
                    f.write("\n".join(label_lines) + ("\n" if label_lines else ""))
                if not label_lines:
                    empty_frames += 1
                if (i + 1) % 50 == 0:
                    unreal.log(f"    Progress: {i + 1}/{self.current_total_samples}")

        unreal.log(f"  Labels: {total_annotations} annotations, "
                   f"{empty_frames} empty frames out of {self.current_total_samples}")

    def _generate_labels_segment(self, target, obj_config, obj_name):
        """Segment mode: differential mask via SceneCapture (requires cv2)."""
        if not HAS_CV2:
            unreal.log_error("  Segmentation mode requires cv2!")
            return

        unreal.log(f"  Generating labels (mask-based, segment)...")
        capture_actor, render_target = setup_scene_capture(self.camera)
        if not capture_actor:
            unreal.log_warning("  SceneCapture failed — cannot generate segment labels.")
            return

        total_annotations = 0
        empty_frames = 0

        with unreal.ScopedSlowTask(self.current_total_samples,
                                    f"Labels: {obj_name}") as slow_task:
            slow_task.make_dialog(True)
            for data in self.current_sample_data:
                if slow_task.should_cancel():
                    break
                slow_task.enter_progress_frame(1)
                i = data["frame_idx"]
                label_path = os.path.join(self.staging_labels, f"{i:06d}.txt")

                if data["is_negative"]:
                    with open(label_path, 'w') as f:
                        f.write("")
                    empty_frames += 1
                    continue

                cam_pos, cam_rot = data["cam_pos"], data["cam_rot"]
                self.camera.set_actor_location_and_rotation(cam_pos, cam_rot, False, True)
                label_lines = []

                actors_to_label = [(self.current_class_map[obj_name], target, obj_name)]
                for co_name, co_actor in self.current_co_visible:
                    actors_to_label.append((self.current_class_map[co_name], co_actor, co_name))

                for class_id, label_actor, label_name in actors_to_label:
                    mask = capture_differential_mask(
                        capture_actor, render_target, self.camera,
                        cam_pos, cam_rot, label_actor)
                    if mask is not None:
                        polys = extract_polygons_from_mask(mask)
                        if polys:
                            for poly in polys:
                                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                                label_lines.append(f"{class_id} {coords}")
                                total_annotations += 1

                with open(label_path, 'w') as f:
                    f.write("\n".join(label_lines) + ("\n" if label_lines else ""))
                if not label_lines:
                    empty_frames += 1
                if (i + 1) % 50 == 0:
                    unreal.log(f"    Progress: {i + 1}/{self.current_total_samples}")

        capture_actor.destroy_actor()
        if unreal.EditorAssetLibrary.does_asset_exist(RT_ASSET_PATH):
            unreal.EditorAssetLibrary.delete_asset(RT_ASSET_PATH)

        unreal.log(f"  Labels: {total_annotations} annotations, "
                   f"{empty_frames} empty frames out of {self.current_total_samples}")

    # -------------------------------------------------------------------------
    # MRQ Rendering
    # -------------------------------------------------------------------------

    def _render(self, sequence, obj_name, obj_config):
        global global_executor
        unreal.log(f"  Starting MRQ render for '{obj_name}'...")

        mrq = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        queue = mrq.get_queue()
        queue.delete_all_jobs()

        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
        job.map = unreal.SoftObjectPath(get_world().get_path_name())
        job.job_name = f"YOLO_V3_{obj_name}"

        config = job.get_configuration()
        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)

        output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        output.output_directory = unreal.DirectoryPath(self.staging_images)
        output.output_resolution = unreal.IntPoint(RESOLUTION_X, RESOLUTION_Y)
        output.file_name_format = "{frame_number}"
        output.zero_pad_frame_numbers = 6
        output.flush_disk_writes_per_shot = True
        output.use_custom_playback_range = True
        output.custom_start_frame = 0
        output.custom_end_frame = self.current_total_samples
        output.handle_frame_count = 0

        aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
        aa.spatial_sample_count = SPATIAL_SAMPLES
        aa.temporal_sample_count = TEMPORAL_SAMPLES
        aa.override_anti_aliasing = True
        aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_FXAA
        aa.render_warm_up_count = WARMUP_FRAMES
        aa.engine_warm_up_count = WARMUP_FRAMES
        aa.render_warm_up_frames = True

        game = config.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
        game.cinematic_quality_settings = True
        game.texture_streaming = unreal.MoviePipelineTextureStreamingMethod.DISABLED
        game.use_lod_zero = True
        game.disable_hlods = True

        console = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
        console.start_console_commands = [
            "r.TemporalAA 0",
            "r.TemporalAA.Quality 0",
            "r.TemporalAACurrentFrameWeight 1.0",
            "r.TemporalAASamples 1",
            "r.TemporalAAFilterSize 0",
            "r.TSR.History.ScreenPercentage 100",
            "r.TSR.History.UpdatePersistentFeedback 0",
            "r.TSR.ShadingRejection.Flickering 0",
            "r.MotionBlurQuality 0",
            "r.MotionBlur.Max 0",
            "r.DefaultFeature.MotionBlur 0",
            "r.SSR.Temporal 0",
            "r.DOF.TemporalAAQuality 0",
            "r.DepthOfFieldQuality 0",
            "r.DOF.Kernel.MaxForegroundRadius 0",
            "r.DOF.Kernel.MaxBackgroundRadius 0",
            "r.DepthOfField.MaxSize 0",
            "ShowFlag.DepthOfField 0",
            "r.DefaultFeature.AntiAliasing 1",
            "r.VolumetricFog.TemporalReprojection 0",
            "r.ScreenPercentage 100",
        ]
        config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)

        # Capture state for the closure
        output_dir = self.current_output_dir
        total_samples = self.current_total_samples
        val_split = obj_config.get("val_split", 0.2)
        class_name = obj_name
        class_map = dict(self.current_class_map)
        generator = self

        global_executor = unreal.MoviePipelinePIEExecutor()

        def on_finished(executor, success):
            global global_executor
            unreal.log("=" * 60)
            unreal.log(f"RENDER COMPLETE: '{class_name}' — Success: {success}")
            flatten_and_renumber_frames(output_dir)
            split_dataset(output_dir, val_split)
            generate_data_yaml(output_dir, class_map, YOLO_V3_MODE)
            unreal.log(f"  Output: {output_dir}")
            unreal.log("=" * 60)
            generator.objects_completed.append(class_name)
            generator._restore_non_targets()
            global_executor = None
            generator._process_next_object()

        global_executor.on_executor_finished_delegate.add_callable(on_finished)
        mrq.render_queue_with_executor_instance(global_executor)

    # -------------------------------------------------------------------------
    # Completion
    # -------------------------------------------------------------------------

    def _on_all_complete(self):
        self._restore_all_initial_transforms()
        unreal.log("=" * 60)
        unreal.log("ALL OBJECTS COMPLETE!")
        unreal.log(f"  Generated {len(self.objects_completed)} dataset(s): {self.objects_completed}")
        unreal.log(f"  Output root: {YOLO_V3_OUTPUT_ROOT}")
        unreal.log("  Run merge_datasets.py to combine into multi-class dataset.")
        unreal.log("=" * 60)


# =============================================================================
# SCENE RESTORE (crash recovery)
# =============================================================================

def restore_scene():
    """Restore all actors to their pre-generation positions using the saved
    transforms file. Call this from the UE5 Python console if the generator
    was interrupted or crashed mid-run.

    Usage:
        from yolo_v3.generate_yolo_v3 import restore_scene
        restore_scene()
    """
    import importlib, sys
    if 'config' in sys.modules:
        importlib.reload(sys.modules['config'])
    import config as cfg
    output_root = getattr(cfg, 'YOLO_V3_OUTPUT_ROOT', '')
    target_tag = getattr(cfg, 'TARGET_TAG', 'TrainObject')
    hide_tag = getattr(cfg, 'HIDE_IN_NEGATIVE_TAG', 'HideInNegative')

    path = os.path.join(output_root, "_initial_transforms.json")
    if not os.path.exists(path):
        unreal.log_warning("No _initial_transforms.json found — nothing to restore.")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    restored = 0
    for actor in subsys.get_all_level_actors():
        label = actor.get_actor_label()
        if label in data:
            t = data[label]
            loc = unreal.Vector(t["x"], t["y"], t["z"])
            rot = unreal.Rotator(t["roll"], t["pitch"], t["yaw"])
            actor.set_actor_location_and_rotation(loc, rot, False, True)
            restored += 1
            unreal.log(f"  Restored '{label}' → ({t['x']:.1f}, {t['y']:.1f}, {t['z']:.1f})")

    if restored:
        os.remove(path)
        unreal.log(f"Restore complete: {restored} actor(s) returned to initial positions.")
    else:
        unreal.log_warning("No matching actors found in scene.")


# =============================================================================
# POST-PROCESSING FUNCTIONS
# =============================================================================

def flatten_and_renumber_frames(output_dir):
    """Renumber frames sequentially and flatten any MRQ subdirectories."""
    images_folder = os.path.join(output_dir, "_staging", "images")
    png_files = sorted(glob.glob(os.path.join(images_folder, "**", "*.png"), recursive=True))
    renamed = 0
    for idx, png_path in enumerate(png_files):
        new_path = os.path.join(images_folder, f"{idx:06d}.png")
        if png_path != new_path:
            os.rename(png_path, new_path)
            renamed += 1
    for item in os.listdir(images_folder):
        item_path = os.path.join(images_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
    unreal.log(f"  Cleanup: renamed {renamed} frames")


def split_dataset(output_dir, val_ratio=0.2):
    staging_images = os.path.join(output_dir, "_staging", "images")
    staging_labels = os.path.join(output_dir, "_staging", "labels")
    all_images = sorted(glob.glob(os.path.join(staging_images, "*.png")))
    if not all_images:
        unreal.log_warning("  No images found for split!")
        return
    random.shuffle(all_images)
    split_idx = max(1, int(len(all_images) * (1 - val_ratio)))
    splits = {"train": all_images[:split_idx], "val": all_images[split_idx:]}
    for split_name, img_list in splits.items():
        img_dir = os.path.join(output_dir, split_name, "images")
        lbl_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for img_path in img_list:
            base = os.path.splitext(os.path.basename(img_path))[0]
            shutil.move(img_path, os.path.join(img_dir, f"{base}.png"))
            lbl = os.path.join(staging_labels, f"{base}.txt")
            if os.path.exists(lbl):
                shutil.move(lbl, os.path.join(lbl_dir, f"{base}.txt"))
        unreal.log(f"  {split_name}: {len(img_list)} samples")
    staging = os.path.join(output_dir, "_staging")
    if os.path.exists(staging):
        shutil.rmtree(staging)


def generate_data_yaml(output_dir, class_map, mode="detect"):
    """Generate data.yaml. class_map is {name: id} dict (supports multi-class via co_visible)."""
    yaml_path = os.path.join(output_dir, "data.yaml")
    task_str = "segment" if mode == "segment" else "detect"
    nc = len(class_map)
    # Invert: {id: name}
    id_to_name = {v: k for k, v in class_map.items()}
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.rstrip('/').rstrip(chr(92))}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write(f"task: {task_str}\n")
        f.write(f"nc: {nc}\n")
        f.write("names:\n")
        for idx in sorted(id_to_name.keys()):
            f.write(f"  {idx}: {id_to_name[idx]}\n")
    unreal.log(f"  data.yaml: nc={nc}, classes={id_to_name}, task={task_str}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if 'yolo_v3_gen' in dir():
    del yolo_v3_gen

if 'global_executor' in dir() and global_executor:
    global_executor = None

yolo_v3_gen = YOLOv3DatasetGenerator()
