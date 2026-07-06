"""
UE5 TAC Challenge — Fast Segmentation Dataset Generator (analytic, no masking)

Sibling of yolo_v3/generate_yolo_v3.py for the TAC Challenge underwater-ocean
scene. The ocean scene has a clear line of sight: nothing ever occludes the
camera→target. That assumption lets us drop the SceneCapture2D two-pass
differential mask entirely and produce segmentation polygons by ANALYTIC
PROJECTION:

    project the object's annotation volume → 2D, take the convex hull,
    clip to the image rectangle → YOLO segment polygon.

Result: no GPU captures, no per-pixel render-target reads, no cv2 dependency
for labels. Each object still produces a standalone YOLO-seg dataset folder
(class 0 = that object). Reuse yolo_v3/merge_datasets.py to combine classes.

This generator runs INSIDE the UE5 Editor Python environment (needs `unreal`).
cv2 / numpy are optional — only used for the downsample + JPG post-processing.

Output structure:
    TAC_SEG_OUTPUT_ROOT/
    └── ocean/
        └── pipeline/
            ├── data.yaml          (task: segment, nc: 1)
            ├── train/images & labels/
            └── val/images & labels/

Run:
    py "C:/.../UE5_SyntheticDataGen/tac_seg/generate_tac_seg.py"
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
    _script_dir = next((p for p in [os.getcwd()] + sys.path
                        if os.path.isfile(os.path.join(p, 'tac_registry.py'))), '')
if _script_dir and _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
# config.py lives in the repo root (parent of tac_seg/)
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir and _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import config as _config_mod
importlib.reload(_config_mod)
import tac_registry as _registry_mod
importlib.reload(_registry_mod)

from config import (
    TARGET_TAG, CAMERA_TAG, IGNORE_TAG,
    SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM, FOCAL_LENGTH_MM,
    RESOLUTION_X, RESOLUTION_Y, WARMUP_FRAMES, SPATIAL_SAMPLES, TEMPORAL_SAMPLES,
)
from tac_registry import get_object_config, resolve_targets

# --- TAC-specific settings (fall back to shared/general defaults) ------------
TAC_SEG_GENERATE = getattr(_config_mod, "TAC_SEG_GENERATE", ["pipeline"])
TAC_SEG_OUTPUT_ROOT = getattr(_config_mod, "TAC_SEG_OUTPUT_ROOT", "C:/UE5_TAC_Seg_Data/")
TAC_SEG_SEQUENCE_PREFIX = getattr(_config_mod, "TAC_SEG_SEQUENCE_PREFIX", "/Game/Generated/TACSeg")
# Ocean is open water — fall back to shared POOL_BOUNDS if a TAC volume is unset.
OCEAN_BOUNDS = getattr(_config_mod, "TAC_OCEAN_BOUNDS",
                       getattr(_config_mod, "POOL_BOUNDS", None))
if OCEAN_BOUNDS is None:
    OCEAN_BOUNDS = {"x_min": -2000.0, "x_max": 2000.0,
                    "y_min": -2000.0, "y_max": 2000.0,
                    "z_min": -2000.0, "z_max": -1000.0}

TAC_SEG_MIN_BBOX_WIDTH_PX = int(getattr(_config_mod, "TAC_SEG_MIN_BBOX_WIDTH_PX", 4))
TAC_SEG_MIN_BBOX_HEIGHT_PX = int(getattr(_config_mod, "TAC_SEG_MIN_BBOX_HEIGHT_PX", 8))
TAC_SEG_MAX_COLLISION_RETRIES = int(getattr(_config_mod, "TAC_SEG_MAX_COLLISION_RETRIES", 20))
TAC_SEG_OBJECT_MIN_SEPARATION = float(getattr(_config_mod, "TAC_SEG_OBJECT_MIN_SEPARATION", 1.0))

TAC_SEG_DOWNSAMPLE_TO = getattr(_config_mod, "TAC_SEG_DOWNSAMPLE_TO", None)
TAC_SEG_IMAGE_FORMAT = str(getattr(_config_mod, "TAC_SEG_IMAGE_FORMAT", "jpg")).lower().lstrip(".")
if TAC_SEG_IMAGE_FORMAT == "jpeg":
    TAC_SEG_IMAGE_FORMAT = "jpg"
if TAC_SEG_IMAGE_FORMAT not in {"png", "jpg"}:
    TAC_SEG_IMAGE_FORMAT = "png"
TAC_SEG_JPEG_QUALITY = max(1, min(100, int(getattr(_config_mod, "TAC_SEG_JPEG_QUALITY", 92))))
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

SEQUENCE_PATH = TAC_SEG_SEQUENCE_PREFIX + "Sequence"
NEAR_Z = 1.0  # camera-space near plane (cm)

# Edge connectivity of an 8-corner box (corner order matches the sx/sy/sz loop).
BOX_EDGE_INDICES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
)

# Global reference to prevent garbage collection during async MRQ render.
global_executor = None


# =============================================================================
# WORLD / MATH HELPERS
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


def _world_to_camera_cv(world_pt, cam_transform):
    """UE world point → OpenCV camera coords (x-right, y-down, z-forward)."""
    cam_inv = cam_transform.inverse()
    local = unreal.MathLibrary.transform_location(cam_inv, world_pt)
    return local.y, -local.z, local.x


def _project_camera_point(cv_x, cv_y, cv_z, intrinsics):
    if cv_z <= 0:
        return [-9999.0, -9999.0]
    u = (cv_x * intrinsics["fx"] / cv_z) + intrinsics["cx"]
    v = (cv_y * intrinsics["fy"] / cv_z) + intrinsics["cy"]
    return [u, v]


def project_point(world_pt, cam_transform, intrinsics):
    return _project_camera_point(*_world_to_camera_cv(world_pt, cam_transform), intrinsics)


def _clamp_to_bounds(pos):
    return unreal.Vector(
        max(OCEAN_BOUNDS["x_min"], min(pos.x, OCEAN_BOUNDS["x_max"])),
        max(OCEAN_BOUNDS["y_min"], min(pos.y, OCEAN_BOUNDS["y_max"])),
        max(OCEAN_BOUNDS["z_min"], min(pos.z, OCEAN_BOUNDS["z_max"])),
    )


def _vec_add(a, b):
    return unreal.Vector(a.x + b.x, a.y + b.y, a.z + b.z)


def _vec_sub(a, b):
    return unreal.Vector(a.x - b.x, a.y - b.y, a.z - b.z)


def _rotate_vector(rot, vec):
    return unreal.MathLibrary.transform_direction(unreal.Transform(rotation=rot), vec)


def _clamp01(v):
    return max(0.0, min(1.0, v))


# =============================================================================
# CAMERA POSITION GENERATORS
# =============================================================================

def generate_vertical_hemisphere(center, min_dist, max_dist, phi_max=90.0):
    """Camera orbits ABOVE — bird's eye to side views."""
    dist = random.uniform(min_dist, max_dist)
    theta = random.uniform(0, 2 * math.pi)
    cos_limit = math.cos(math.radians(phi_max))
    phi = math.acos(random.uniform(cos_limit, 1))
    dx = dist * math.sin(phi) * math.cos(theta)
    dy = dist * math.sin(phi) * math.sin(theta)
    dz = dist * math.cos(phi)
    return _clamp_to_bounds(unreal.Vector(center.x + dx, center.y + dy, center.z + dz))


def generate_horizontal_hemisphere(center, min_dist, max_dist, theta_range=None):
    """Camera orbits AROUND at eye level. 20% chance of looking up."""
    dist = random.uniform(min_dist, max_dist)
    if theta_range is not None:
        theta = math.radians(random.uniform(theta_range[0], theta_range[1]))
    else:
        theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(random.uniform(0, 0.5))
    dx = dist * math.sin(phi) * math.cos(theta)
    dy = dist * math.sin(phi) * math.sin(theta)
    dz = dist * math.cos(phi)
    if random.random() < 0.2:
        dz = -dz
    return _clamp_to_bounds(unreal.Vector(center.x + dx, center.y + dy, center.z + dz))


def generate_camera_position(center, obj_config):
    min_d, max_d = obj_config["min_distance"], obj_config["max_distance"]
    if obj_config["hemisphere"] == "horizontal":
        return generate_horizontal_hemisphere(
            center, min_d, max_d, theta_range=obj_config.get("theta_range"))
    return generate_vertical_hemisphere(
        center, min_d, max_d, phi_max=obj_config.get("phi_max", 90.0))


# =============================================================================
# ANNOTATION GEOMETRY
# =============================================================================

def _get_annotation_obb(actor):
    """Return (center_world, axes, half_extents) for the annotation volume.

    axes is a list of 3 unit unreal.Vector (the box's local X/Y/Z in world
    space); half_extents is (hx, hy, hz). Priority:
      1. BoxComponent tagged `DOPE_Bounds`
      2. single visible StaticMeshComponent's mesh bounds
      3. actor AABB fallback (axes = world axes)
    """
    for comp in actor.get_components_by_class(unreal.BoxComponent):
        if comp.component_has_tag("DOPE_Bounds"):
            extent = comp.get_unscaled_box_extent() * comp.get_world_scale()
            rot = comp.get_world_rotation()
            axes = [_rotate_vector(rot, unreal.Vector(1, 0, 0)),
                    _rotate_vector(rot, unreal.Vector(0, 1, 0)),
                    _rotate_vector(rot, unreal.Vector(0, 0, 1))]
            return comp.get_world_location(), axes, (extent.x, extent.y, extent.z)

    mesh_comps = [c for c in actor.get_components_by_class(unreal.StaticMeshComponent)
                  if c.static_mesh and c.is_visible()]
    if len(mesh_comps) == 1:
        comp = mesh_comps[0]
        b = comp.static_mesh.get_bounds()
        scale = comp.get_world_scale()
        rot = comp.get_world_rotation()
        center_world = unreal.MathLibrary.transform_location(comp.get_world_transform(), b.origin)
        axes = [_rotate_vector(rot, unreal.Vector(1, 0, 0)),
                _rotate_vector(rot, unreal.Vector(0, 1, 0)),
                _rotate_vector(rot, unreal.Vector(0, 0, 1))]
        ext = b.box_extent
        return center_world, axes, (ext.x * scale.x, ext.y * scale.y, ext.z * scale.z)

    origin, extent = actor.get_actor_bounds(False)
    axes = [unreal.Vector(1, 0, 0), unreal.Vector(0, 1, 0), unreal.Vector(0, 0, 1)]
    return origin, axes, (extent.x, extent.y, extent.z)


def _obb_corners(center, axes, half):
    """8 world-space corners of an oriented box."""
    corners = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                corners.append(_vec_add(center, unreal.Vector(
                    axes[0].x * sx * half[0] + axes[1].x * sy * half[1] + axes[2].x * sz * half[2],
                    axes[0].y * sx * half[0] + axes[1].y * sy * half[1] + axes[2].y * sz * half[2],
                    axes[0].z * sx * half[0] + axes[1].z * sy * half[1] + axes[2].z * sz * half[2],
                )))
    return corners


def _cylinder_silhouette_points(center, axes, half, segments=16):
    """World points on a pipe-like cylinder derived from an OBB.

    The longest box axis becomes the cylinder axis; the cross-section radius
    circumscribes the other two half-extents so the hull fully contains the
    pipe (convex over-cover of the rounded caps). Rim points at both ends give
    a capsule silhouette that hugs a round pipe far better than 8 box corners.
    """
    axis_idx = max(range(3), key=lambda i: half[i])
    perp_idx = [i for i in range(3) if i != axis_idx]
    axis = axes[axis_idx]
    p1, p2 = axes[perp_idx[0]], axes[perp_idx[1]]
    half_len = half[axis_idx]
    radius = max(half[perp_idx[0]], half[perp_idx[1]])

    pts = []
    for end in (half_len, -half_len):
        base = _vec_add(center, unreal.Vector(axis.x * end, axis.y * end, axis.z * end))
        for k in range(segments):
            a = 2.0 * math.pi * k / segments
            c, s = math.cos(a) * radius, math.sin(a) * radius
            pts.append(_vec_add(base, unreal.Vector(
                p1.x * c + p2.x * s, p1.y * c + p2.y * s, p1.z * c + p2.z * s)))
    return pts


def _silhouette_world_points(actor, cfg):
    """World points whose 2D convex hull approximates the segment silhouette."""
    center, axes, half = _get_annotation_obb(actor)
    if cfg.get("silhouette") == "cylinder":
        segments = int(cfg.get("silhouette_opts", {}).get("segments", 16))
        return _cylinder_silhouette_points(center, axes, half, segments=segments)
    return _obb_corners(center, axes, half)


def _get_annotation_center(actor):
    center, _, _ = _get_annotation_obb(actor)
    return center


def _describe_annotation_source(actor):
    for comp in actor.get_components_by_class(unreal.BoxComponent):
        if comp.component_has_tag("DOPE_Bounds"):
            return f"DOPE_Bounds BoxComponent '{comp.get_name()}'"
    mesh_comps = [c for c in actor.get_components_by_class(unreal.StaticMeshComponent)
                  if c.static_mesh and c.is_visible()]
    if len(mesh_comps) == 1:
        return f"StaticMeshComponent mesh bounds '{mesh_comps[0].static_mesh.get_name()}'"
    if len(mesh_comps) > 1:
        return f"actor AABB fallback ({len(mesh_comps)} mesh comps)"
    return "actor AABB fallback"


def _get_bottom_pivot_local_offset(actor, actor_loc, actor_rot):
    """Annotation-box bottom-center expressed in actor-local space."""
    center, axes, half = _get_annotation_obb(actor)
    corners = _obb_corners(center, axes, half)
    actor_tf_inv = unreal.Transform(location=actor_loc, rotation=actor_rot).inverse()
    local = [unreal.MathLibrary.transform_location(actor_tf_inv, c) for c in corners]
    xs = [c.x for c in local]
    ys = [c.y for c in local]
    zs = [c.z for c in local]
    return unreal.Vector((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, min(zs))


# =============================================================================
# 2D PROJECTION / CONVEX-HULL / SCREEN-CLIP
# =============================================================================

def _cross_2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points):
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull if len(hull) >= 3 else pts


def _ensure_ccw(poly):
    if len(poly) < 3:
        return poly
    signed = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        signed += (x2 - x1) * (y2 + y1)
    return list(reversed(poly)) if signed > 0 else poly


def _line_intersection(p1, p2, q1, q2):
    a1, b1 = p2[1] - p1[1], p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]
    a2, b2 = q2[1] - q1[1], q1[0] - q2[0]
    c2 = a2 * q1[0] + b2 * q1[1]
    det = (a1 * b2) - (a2 * b1)
    if abs(det) < 1e-9:
        return p2
    return (((b2 * c1) - (b1 * c2)) / det, ((a1 * c2) - (a2 * c1)) / det)


def _clip_polygon_to_rect(poly, x_max, y_max):
    """Sutherland-Hodgman clip of a convex polygon to [0,x_max]x[0,y_max]."""
    clip = _ensure_ccw([(0.0, 0.0), (x_max, 0.0), (x_max, y_max), (0.0, y_max)])
    output = _ensure_ccw(list(poly))
    if len(output) < 3:
        return []
    for i in range(len(clip)):
        cp1, cp2 = clip[i], clip[(i + 1) % len(clip)]
        inp, output = output, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            e_in = _cross_2d(cp1, cp2, e) >= 0
            s_in = _cross_2d(cp1, cp2, s) >= 0
            if e_in:
                if not s_in:
                    output.append(_line_intersection(s, e, cp1, cp2))
                output.append(e)
            elif s_in:
                output.append(_line_intersection(s, e, cp1, cp2))
            s = e
    return output


def _project_segment(actor, cam_tf, intrinsics, cfg):
    """Analytic segmentation. Returns (polygon_norm, bbox_norm) or None.

    polygon_norm: list of (x, y) in [0,1] (image-normalized) — the convex hull
                  of the projected silhouette points, clipped to the image.
    bbox_norm:    (xc, yc, w, h) normalized, derived from the polygon.
    """
    world_pts = _silhouette_world_points(actor, cfg)
    screen_pts = []
    for wp in world_pts:
        cv_x, cv_y, cv_z = _world_to_camera_cv(wp, cam_tf)
        if cv_z < NEAR_Z:
            continue  # behind / too close to camera — drop
        screen_pts.append(tuple(_project_camera_point(cv_x, cv_y, cv_z, intrinsics)))

    if len(screen_pts) < 3:
        return None

    hull = _convex_hull(screen_pts)
    clipped = _clip_polygon_to_rect(hull, float(RESOLUTION_X), float(RESOLUTION_Y))
    if len(clipped) < 3:
        return None

    xs = [p[0] for p in clipped]
    ys = [p[1] for p in clipped]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None

    poly = [(_clamp01(x / RESOLUTION_X), _clamp01(y / RESOLUTION_Y)) for x, y in clipped]
    bbox = (
        _clamp01(((x1 + x2) / 2.0) / RESOLUTION_X),
        _clamp01(((y1 + y2) / 2.0) / RESOLUTION_Y),
        _clamp01((x2 - x1) / RESOLUTION_X),
        _clamp01((y2 - y1) / RESOLUTION_Y),
    )
    return poly, bbox


def _bbox_meets_min_size(bbox):
    if not bbox:
        return False
    _, _, w, h = bbox
    return (w * RESOLUTION_X >= TAC_SEG_MIN_BBOX_WIDTH_PX and
            h * RESOLUTION_Y >= TAC_SEG_MIN_BBOX_HEIGHT_PX)


def _bbox_touches_edge(bbox, margin_px=1.0):
    if not bbox:
        return False
    xc, yc, w, h = bbox
    mx, my = margin_px / RESOLUTION_X, margin_px / RESOLUTION_Y
    return ((xc - w / 2.0) <= mx or (xc + w / 2.0) >= 1.0 - mx or
            (yc - h / 2.0) <= my or (yc + h / 2.0) >= 1.0 - my)


def _actor_xy_radius(actor):
    _, extent = actor.get_actor_bounds(False)
    return math.sqrt(extent.x * extent.x + extent.y * extent.y)


def _check_no_overlap_2d(entries, min_sep):
    n = len(entries)
    for i in range(n):
        for j in range(i + 1, n):
            la, ra = entries[i]
            lb, rb = entries[j]
            dist = math.sqrt((la.x - lb.x) ** 2 + (la.y - lb.y) ** 2)
            if dist < (ra + rb + min_sep):
                return False
    return True


# =============================================================================
# MAIN GENERATOR
# =============================================================================

class TACSegGenerator:
    def __init__(self):
        self.camera = None
        self.all_target_actors = []
        self.intrinsics = calculate_intrinsics()
        self.object_queue = []
        self.objects_completed = []
        self.initial_transforms = {}

        # per-object state
        self.current_target = None
        self.current_config = None
        self.current_output_dir = ""
        self.current_sample_data = []
        self.current_total_samples = 0
        self.staging_images = ""
        self.staging_labels = ""
        self.non_target_original_locs = {}
        self.current_co_visible = []
        self.current_class_map = {}
        self.current_class_name_map = {}
        self.current_sub_actors = []
        self.current_orbit_anchor = None

        unreal.log("=" * 60)
        unreal.log("UE5 TAC SEG GENERATOR — analytic (no SceneCapture)")
        unreal.log("=" * 60)

        self._find_actors()
        if not self.camera:
            unreal.log_error(f"No camera with tag '{CAMERA_TAG}' found!")
            return

        self._snapshot_initial_transforms()
        self._configure_camera()

        try:
            self.object_queue = resolve_targets(TAC_SEG_GENERATE)
        except ValueError as e:
            unreal.log_error(f"Invalid TAC_SEG_GENERATE config: {e}")
            return

        unreal.log(f"Objects ({len(self.object_queue)}): {self.object_queue}")
        unreal.log(f"Output: {TAC_SEG_OUTPUT_ROOT}")
        self._process_next_object()

    # ----- discovery ---------------------------------------------------------

    def _find_actors(self):
        subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        ignored = 0
        for actor in subsys.get_all_level_actors():
            if actor.actor_has_tag(TARGET_TAG):
                self.all_target_actors.append(actor)
            elif actor.actor_has_tag(CAMERA_TAG):
                self.camera = actor
            elif actor.actor_has_tag(IGNORE_TAG):
                actor.destroy_actor()
                ignored += 1
        if ignored:
            unreal.log(f"  Removed {ignored} ignored object(s)")
        unreal.log(f"  Found {len(self.all_target_actors)} target(s), "
                   f"camera: {'Yes' if self.camera else 'No'}")

    def _configure_camera(self):
        cc = self.camera.get_cine_camera_component()
        cc.filmback.sensor_width = SENSOR_WIDTH_MM
        cc.filmback.sensor_height = SENSOR_HEIGHT_MM
        cc.current_focal_length = FOCAL_LENGTH_MM
        cc.focus_settings.focus_method = unreal.CameraFocusMethod.DISABLE

    def _snapshot_initial_transforms(self):
        self.initial_transforms = {}
        for actor in self.all_target_actors:
            self.initial_transforms[actor] = (
                actor.get_actor_location(), actor.get_actor_rotation())
        self._save_transforms_to_disk()
        unreal.log(f"  Saved initial transforms for {len(self.initial_transforms)} actor(s)")

    def _save_transforms_to_disk(self):
        data = {}
        for actor, (loc, rot) in self.initial_transforms.items():
            data[actor.get_actor_label()] = {
                "x": loc.x, "y": loc.y, "z": loc.z,
                "roll": rot.roll, "pitch": rot.pitch, "yaw": rot.yaw,
            }
        os.makedirs(TAC_SEG_OUTPUT_ROOT, exist_ok=True)
        with open(os.path.join(TAC_SEG_OUTPUT_ROOT, "_initial_transforms.json"), 'w') as f:
            json.dump(data, f, indent=2)

    def _apply_initial_transforms(self):
        count = 0
        for actor, (loc, rot) in self.initial_transforms.items():
            try:
                actor.set_actor_location_and_rotation(loc, rot, False, True)
                count += 1
            except Exception:
                pass
        return count

    def _restore_all_initial_transforms(self):
        count = self._apply_initial_transforms()
        path = os.path.join(TAC_SEG_OUTPUT_ROOT, "_initial_transforms.json")
        if os.path.exists(path):
            os.remove(path)
        unreal.log(f"  Restored {count} actor(s) to initial transforms")

    def _find_actor_by_label(self, label):
        for actor in self.all_target_actors:
            if actor.get_actor_label() == label:
                return actor
        return None

    # ----- per-object loop ---------------------------------------------------

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
                   f"Silhouette: {obj_config.get('silhouette')} | "
                   f"Dist: {obj_config['min_distance']}-{obj_config['max_distance']}cm")
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
        self.current_orbit_anchor = None

        anchor_label = obj_config.get("orbit_anchor_label")
        if anchor_label:
            anchor = self._find_actor_by_label(anchor_label)
            if anchor:
                self.current_orbit_anchor = anchor
                unreal.log(f"  Orbit anchor: '{anchor_label}'")
            else:
                unreal.log_warning(f"  orbit_anchor_label '{anchor_label}' not found")

        # co_visible: each becomes its own class
        self.current_co_visible = []
        for co_name in obj_config.get("co_visible", []):
            try:
                co_cfg = get_object_config(co_name)
            except KeyError:
                unreal.log_warning(f"  Co-visible '{co_name}' not in registry")
                continue
            co_actor = self._find_actor_by_label(co_cfg["actor_label"])
            if co_actor:
                self.current_co_visible.append((co_name, co_actor))
            else:
                unreal.log_warning(f"  Co-visible actor '{co_name}' not found in scene")

        # class map: target=0, co_visible sorted alphabetically = 1,2,...
        target_class_name = obj_config.get("class_name", obj_name)
        self.current_class_map = {obj_name: 0}
        self.current_class_name_map = {0: target_class_name}
        for i, (co_name, _) in enumerate(sorted(self.current_co_visible)):
            co_id = i + 1
            self.current_class_map[co_name] = co_id
            try:
                co_cfg = get_object_config(co_name)
                self.current_class_name_map[co_id] = co_cfg.get("class_name", co_name)
            except KeyError:
                self.current_class_name_map[co_id] = co_name
        if self.current_co_visible:
            unreal.log(f"  Co-visible class map: {self.current_class_map}")

        # sub_actors: same class as the target
        self.current_sub_actors = []
        for sub_label in obj_config.get("sub_actors", []):
            sub_actor = self._find_actor_by_label(sub_label)
            if sub_actor:
                self.current_sub_actors.append((obj_name, sub_actor))
            else:
                unreal.log_warning(f"  Sub-actor '{sub_label}' not found in scene")
        for co_name, _ in self.current_co_visible:
            try:
                co_cfg = get_object_config(co_name)
            except KeyError:
                continue
            for sub_label in co_cfg.get("sub_actors", []):
                sub_actor = self._find_actor_by_label(sub_label)
                if sub_actor:
                    self.current_sub_actors.append((co_name, sub_actor))
        if self.current_sub_actors:
            unreal.log(f"  Sub-actors: {[(k, a.get_actor_label()) for k, a in self.current_sub_actors]}")

        cam_group = obj_config["camera_group"]
        self.current_output_dir = os.path.join(TAC_SEG_OUTPUT_ROOT, cam_group, obj_name)
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

        self._generate_labels_segment(target, obj_config, obj_name)
        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
        self._render(sequence, obj_name, obj_config)

    def _hide_non_targets(self, target_actor):
        keep_above = {a for _, a in self.current_co_visible}
        keep_above |= {a for _, a in self.current_sub_actors}
        if self.current_orbit_anchor is not None:
            keep_above.add(self.current_orbit_anchor)
        self.non_target_original_locs = {}
        for actor in self.all_target_actors:
            if actor != target_actor and actor not in keep_above:
                loc = actor.get_actor_location()
                self.non_target_original_locs[actor] = loc
                actor.set_actor_location(unreal.Vector(loc.x, loc.y, -20000.0), False, False)
        if self.non_target_original_locs:
            unreal.log(f"  Hidden {len(self.non_target_original_locs)} non-target actor(s)")

    def _restore_non_targets(self):
        for actor, orig_loc in self.non_target_original_locs.items():
            actor.set_actor_location(orig_loc, False, False)
        self.non_target_original_locs = {}

    # ----- track helpers -----------------------------------------------------

    def _resolve_rotation_dr(self, cfg, apply_to_self=False, apply_to_sub_actors=False):
        rdr = cfg.get("rotation_dr")
        if not rdr:
            return None
        if apply_to_self and not rdr.get("apply_to_self", True):
            return None
        if apply_to_sub_actors and not rdr.get("apply_to_sub_actors", False):
            return None
        return rdr

    def _build_actor_track(self, actor, channels, rotation_dr=None):
        track = {
            "actor": actor,
            "channels": channels,
            "orig_loc": actor.get_actor_location(),
            "orig_rot": actor.get_actor_rotation(),
            "orig_scale": actor.get_actor_scale3d(),
        }
        if rotation_dr and rotation_dr.get("mode", "bottom_pivot") == "bottom_pivot":
            pivot_local = _get_bottom_pivot_local_offset(actor, track["orig_loc"], track["orig_rot"])
            track["rotation_dr"] = {
                "roll_range": rotation_dr.get("roll_range", 0.0),
                "pitch_range": rotation_dr.get("pitch_range", 0.0),
                "pivot_local": pivot_local,
                "pivot_world": _vec_add(track["orig_loc"],
                                        _rotate_vector(track["orig_rot"], pivot_local)),
            }
        return track

    @staticmethod
    def _write_transform_keys(channels, frame_time, loc, rot, scale):
        channels[0].add_key(frame_time, loc.x)
        channels[1].add_key(frame_time, loc.y)
        channels[2].add_key(frame_time, loc.z)
        channels[3].add_key(frame_time, rot.roll)
        channels[4].add_key(frame_time, rot.pitch)
        channels[5].add_key(frame_time, rot.yaw)
        channels[6].add_key(frame_time, scale.x)
        channels[7].add_key(frame_time, scale.y)
        channels[8].add_key(frame_time, scale.z)

    def _write_track_pose(self, track, frame_time, loc=None, rot=None, underground=False):
        if underground:
            loc = unreal.Vector(track["orig_loc"].x, track["orig_loc"].y, -20000.0)
            rot = track["orig_rot"]
        elif loc is None or rot is None:
            loc, rot = track["orig_loc"], track["orig_rot"]
        self._write_transform_keys(track["channels"], frame_time, loc, rot, track["orig_scale"])

    def _sample_track_pose(self, track):
        rdr = track.get("rotation_dr")
        if not rdr:
            return track["orig_loc"], track["orig_rot"]
        rot = unreal.Rotator(
            roll=track["orig_rot"].roll + random.uniform(-rdr["roll_range"], rdr["roll_range"]),
            pitch=track["orig_rot"].pitch + random.uniform(-rdr["pitch_range"], rdr["pitch_range"]),
            yaw=track["orig_rot"].yaw,
        )
        loc = _vec_sub(rdr["pivot_world"], _rotate_vector(rot, rdr["pivot_local"]))
        return loc, rot

    @staticmethod
    def _sample_placement_pose(track):
        placement = track.get("placement")
        if not placement:
            return None
        origin = track["orig_loc"]
        loc = unreal.Vector(
            random.uniform(origin.x - placement["xy_range_x"], origin.x + placement["xy_range_x"]),
            random.uniform(origin.y - placement["xy_range_y"], origin.y + placement["xy_range_y"]),
            origin.z,
        )
        rot = unreal.Rotator(
            roll=random.uniform(-placement["roll_range"], placement["roll_range"]),
            pitch=random.uniform(placement["pitch_min"], placement["pitch_max"]),
            yaw=random.uniform(0.0, placement["yaw_range"]),
        )
        return loc, rot

    def _apply_actor_states(self, actor_states):
        for actor, state in actor_states.items():
            actor.set_actor_location_and_rotation(state["loc"], state["rot"], False, True)

    # ----- sequence ----------------------------------------------------------

    def _create_sequence(self, target, obj_config, obj_name):
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
        if unreal.EditorAssetLibrary.does_asset_exist(SEQUENCE_PATH):
            unreal.EditorAssetLibrary.delete_asset(SEQUENCE_PATH)
        pkg_path, asset_name = SEQUENCE_PATH.rsplit('/', 1)
        seq = asset_tools.create_asset(
            asset_name, pkg_path, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
        seq.set_display_rate(unreal.FrameRate(24, 1))

        total_frames = self.current_total_samples
        cam_binding = seq.add_possessable(self.camera)
        cam_section = cam_binding.add_track(unreal.MovieScene3DTransformTrack).add_section()
        cam_section.set_range(0, total_frames + 10)
        seq.set_playback_start(0)
        seq.set_playback_end(total_frames)
        cam_channels = cam_section.get_all_channels()

        def _new_track(actor, rotation_dr=None, placement=None):
            section = seq.add_possessable(actor).add_track(
                unreal.MovieScene3DTransformTrack).add_section()
            section.set_range(0, total_frames + 10)
            t = self._build_actor_track(actor, section.get_all_channels(), rotation_dr=rotation_dr)
            t["placement"] = placement
            self._write_track_pose(t, unreal.FrameNumber(0))
            return t

        target_track = _new_track(
            target,
            rotation_dr=self._resolve_rotation_dr(obj_config, apply_to_self=True),
            placement=obj_config.get("placement"))
        orig_loc = target_track["orig_loc"]

        co_tracks = []
        for co_name, co_actor in self.current_co_visible:
            co_cfg = get_object_config(co_name)
            co_tracks.append(_new_track(
                co_actor,
                rotation_dr=self._resolve_rotation_dr(co_cfg, apply_to_self=True),
                placement=co_cfg.get("placement")))

        sub_tracks = []
        for sub_key, sub_actor in self.current_sub_actors:
            sub_cfg = obj_config if sub_key == obj_name else get_object_config(sub_key)
            sub_tracks.append(_new_track(
                sub_actor,
                rotation_dr=self._resolve_rotation_dr(sub_cfg, apply_to_sub_actors=True)))

        num_positive = obj_config["samples"]
        num_negative = self.current_total_samples - num_positive
        frame_types = ['positive'] * num_positive + ['negative'] * num_negative
        random.shuffle(frame_types)

        placement = obj_config.get("placement")
        jitter_enabled = obj_config.get("enable_jitter", True)
        jitter_pitch = obj_config.get("jitter_max_pitch", 5.0)
        filter_edges = not obj_config.get("samples_on_edges", True)
        silhouette_cfg = obj_config

        for i in range(self.current_total_samples):
            frame_time = unreal.FrameNumber(i)
            is_negative = (frame_types[i] == 'negative')

            if is_negative:
                self._write_track_pose(target_track, frame_time, underground=True)
                for t in co_tracks + sub_tracks:
                    self._write_track_pose(t, frame_time, underground=True)
                cam_pos = unreal.Vector(
                    random.uniform(OCEAN_BOUNDS["x_min"], OCEAN_BOUNDS["x_max"]),
                    random.uniform(OCEAN_BOUNDS["y_min"], OCEAN_BOUNDS["y_max"]),
                    random.uniform(OCEAN_BOUNDS["z_min"], OCEAN_BOUNDS["z_max"]))
                cam_rot = unreal.Rotator(roll=0.0,
                                         pitch=random.uniform(-70.0, 0.0),
                                         yaw=random.uniform(0.0, 360.0))
                self.current_sample_data.append({
                    "frame_idx": i, "cam_pos": cam_pos, "cam_rot": cam_rot,
                    "is_negative": True})
            else:
                # sample target pose
                if placement:
                    target_loc, target_rot = self._sample_placement_pose(target_track)
                else:
                    target_loc, target_rot = self._sample_track_pose(target_track)
                self._write_track_pose(target_track, frame_time, target_loc, target_rot)

                # camera aimed at anchor (or target)
                orbit_anchor = self.current_orbit_anchor or target
                bbox_center = _get_annotation_center(orbit_anchor)
                orbit_center = orbit_anchor.get_actor_location()
                if placement and orbit_anchor == target:
                    offset = _vec_sub(target_loc, orig_loc)
                    bbox_center = _vec_add(bbox_center, offset)
                    orbit_center = target_loc

                cam_pos = generate_camera_position(orbit_center, obj_config)
                cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, bbox_center)

                if jitter_enabled:
                    dist = math.sqrt((cam_pos.x - bbox_center.x) ** 2 +
                                     (cam_pos.y - bbox_center.y) ** 2 +
                                     (cam_pos.z - bbox_center.z) ** 2)
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

                frame_actor_states = {target: {"loc": target_loc, "rot": target_rot}}
                edge_hidden = set()

                secondary = ([(t, *self._sample_placement_pose(t)) if t.get("placement")
                              else (t, *self._sample_track_pose(t)) for t in co_tracks] +
                             [(t, *self._sample_track_pose(t)) for t in sub_tracks])

                if filter_edges and secondary:
                    cam_tf = unreal.Transform(location=cam_pos, rotation=cam_rot)
                    for t, loc, rot in secondary:
                        actor = t["actor"]
                        actor.set_actor_location_and_rotation(loc, rot, False, True)
                        proj = _project_segment(actor, cam_tf, self.intrinsics, silhouette_cfg)
                        if proj and _bbox_touches_edge(proj[1]):
                            edge_hidden.add(actor)

                for t, loc, rot in secondary:
                    actor = t["actor"]
                    if actor in edge_hidden:
                        self._write_track_pose(t, frame_time, underground=True)
                    else:
                        self._write_track_pose(t, frame_time, loc, rot)
                        frame_actor_states[actor] = {"loc": loc, "rot": rot}

                self.current_sample_data.append({
                    "frame_idx": i, "cam_pos": cam_pos, "cam_rot": cam_rot,
                    "is_negative": False, "actor_states": frame_actor_states,
                    "edge_hidden": edge_hidden})

            cam_channels[0].add_key(frame_time, cam_pos.x)
            cam_channels[1].add_key(frame_time, cam_pos.y)
            cam_channels[2].add_key(frame_time, cam_pos.z)
            cam_channels[3].add_key(frame_time, cam_rot.roll)
            cam_channels[4].add_key(frame_time, cam_rot.pitch)
            cam_channels[5].add_key(frame_time, cam_rot.yaw)

        # constant interpolation on every channel
        all_tracks = [target_track] + co_tracks + sub_tracks
        for ch in cam_channels:
            for key in ch.get_keys():
                key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)
        for t in all_tracks:
            for ch in t["channels"]:
                for key in ch.get_keys():
                    key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)

        camera_cut = self._add_camera_cut_track(seq)
        if camera_cut:
            bid = unreal.MovieSceneObjectBindingID()
            bid.set_editor_property("guid", cam_binding.get_id())
            cs = camera_cut.add_section()
            cs.set_range(0, total_frames + 10)
            cs.set_camera_binding_id(bid)

        unreal.log(f"  Sequence created: {self.current_total_samples} samples")
        return seq

    def _add_camera_cut_track(self, seq):
        try:
            return seq.get_movie_scene().add_camera_cut_track()
        except Exception:
            try:
                return seq.add_track(unreal.MovieSceneCameraCutTrack)
            except Exception:
                return None

    # ----- labels (analytic segmentation) ------------------------------------

    def _generate_labels_segment(self, target, obj_config, obj_name):
        unreal.log("  Generating labels (analytic hull projection)...")
        total_annotations = 0
        empty_frames = 0
        skip_target = obj_config.get("skip_target_bbox", False)

        with unreal.ScopedSlowTask(self.current_total_samples, f"Labels: {obj_name}") as slow_task:
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
                self._apply_actor_states(data.get("actor_states", {}))
                self.camera.set_actor_location_and_rotation(cam_pos, cam_rot, False, True)
                try:
                    cine = self.camera.get_cine_camera_component()
                    cam_tf = unreal.Transform(location=cine.get_world_location(),
                                              rotation=cine.get_world_rotation())
                except (AttributeError, Exception):
                    cam_tf = unreal.Transform(location=cam_pos, rotation=cam_rot)

                edge_hidden = data.get("edge_hidden", set())
                actors_to_label = []
                if not skip_target:
                    actors_to_label.append((self.current_class_map[obj_name], target, obj_name))
                for co_name, co_actor in self.current_co_visible:
                    if co_actor not in edge_hidden:
                        actors_to_label.append((self.current_class_map[co_name], co_actor, co_name))
                for sub_key, sub_actor in self.current_sub_actors:
                    if sub_actor not in edge_hidden:
                        actors_to_label.append((self.current_class_map[sub_key], sub_actor, sub_key))

                label_lines = []
                for class_id, actor, label_name in actors_to_label:
                    cfg = obj_config if label_name == obj_name else get_object_config(label_name)
                    proj = _project_segment(actor, cam_tf, self.intrinsics, cfg)
                    if not proj:
                        continue
                    poly, bbox = proj
                    if not _bbox_meets_min_size(bbox):
                        continue
                    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                    label_lines.append(f"{class_id} {coords}")
                    total_annotations += 1

                with open(label_path, 'w') as f:
                    f.write("\n".join(label_lines) + ("\n" if label_lines else ""))
                if not label_lines:
                    empty_frames += 1
                if (i + 1) % 100 == 0:
                    unreal.log(f"    Progress: {i + 1}/{self.current_total_samples}")

        unreal.log(f"  Labels: {total_annotations} annotations, "
                   f"{empty_frames} empty frames of {self.current_total_samples}")

    # ----- render ------------------------------------------------------------

    def _render(self, sequence, obj_name, obj_config):
        global global_executor
        unreal.log(f"  Starting MRQ render for '{obj_name}'...")

        mrq = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        queue = mrq.get_queue()
        queue.delete_all_jobs()

        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
        job.map = unreal.SoftObjectPath(get_world().get_path_name())
        job.job_name = f"TAC_SEG_{obj_name}"

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
        aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_FXAA  # NEVER change — anti-ghosting
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
            "r.Tonemapper.Sharpen 0",
            "r.MipMapLODBias 0.5",
        ]
        config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)

        output_dir = self.current_output_dir
        val_split = obj_config.get("val_split", 0.2)
        class_name = obj_name
        class_map = dict(self.current_class_map)
        class_name_map = dict(self.current_class_name_map)
        generator = self

        global_executor = unreal.MoviePipelinePIEExecutor()

        def on_finished(executor, success):
            global global_executor
            unreal.log("=" * 60)
            unreal.log(f"RENDER COMPLETE: '{class_name}' — Success: {success}")
            flatten_and_renumber_frames(output_dir)
            split_dataset(output_dir, val_split)
            generate_data_yaml(output_dir, class_map, class_name_map)
            unreal.log(f"  Output: {output_dir}")
            unreal.log("=" * 60)
            generator.objects_completed.append(class_name)
            generator._restore_non_targets()
            restored = generator._apply_initial_transforms()
            unreal.log(f"  Reset {restored} actor pose(s) before next object")
            global_executor = None
            generator._process_next_object()

        global_executor.on_executor_finished_delegate.add_callable(on_finished)
        mrq.render_queue_with_executor_instance(global_executor)

    def _on_all_complete(self):
        self._restore_all_initial_transforms()
        unreal.log("=" * 60)
        unreal.log("ALL OBJECTS COMPLETE!")
        unreal.log(f"  Generated {len(self.objects_completed)} dataset(s): {self.objects_completed}")
        unreal.log(f"  Output root: {TAC_SEG_OUTPUT_ROOT}")
        unreal.log("  Reuse yolo_v3/merge_datasets.py to combine classes.")
        unreal.log("=" * 60)


# =============================================================================
# SCENE RESTORE (crash recovery)
# =============================================================================

def restore_scene():
    """Restore actors to pre-generation positions from the saved JSON.

    Usage in the UE5 Python console:
        from tac_seg.generate_tac_seg import restore_scene
        restore_scene()
    """
    import importlib as _il
    if 'config' in sys.modules:
        _il.reload(sys.modules['config'])
    import config as cfg
    output_root = getattr(cfg, 'TAC_SEG_OUTPUT_ROOT', TAC_SEG_OUTPUT_ROOT)
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
            actor.set_actor_location_and_rotation(
                unreal.Vector(t["x"], t["y"], t["z"]),
                unreal.Rotator(t["roll"], t["pitch"], t["yaw"]), False, True)
            restored += 1
    if restored:
        os.remove(path)
        unreal.log(f"Restore complete: {restored} actor(s).")
    else:
        unreal.log_warning("No matching actors found in scene.")


# =============================================================================
# POST-PROCESSING (cv2 only needed for downsample / JPG)
# =============================================================================

def _iter_image_files(images_folder):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(images_folder, f"*{ext}")))
    return sorted(files)


def flatten_and_renumber_frames(output_dir):
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
    _downsample_images(images_folder)
    _finalize_image_format(images_folder)


def _downsample_images(images_folder):
    if not TAC_SEG_DOWNSAMPLE_TO:
        return
    if not HAS_CV2:
        unreal.log_warning("  Downsample: OpenCV unavailable; keeping full-res frames")
        return
    target_w, target_h = int(TAC_SEG_DOWNSAMPLE_TO[0]), int(TAC_SEG_DOWNSAMPLE_TO[1])
    image_files = _iter_image_files(images_folder)
    if not image_files:
        return
    sample = cv2.imread(image_files[0], cv2.IMREAD_UNCHANGED)
    if sample is None:
        return
    h, w = sample.shape[:2]
    if (w, h) == (target_w, target_h):
        return
    resized = 0
    for p in image_files:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        out = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(p, out)
        resized += 1
    unreal.log(f"  Downsampled {resized} frames: {w}x{h} -> {target_w}x{target_h}")


def _finalize_image_format(images_folder):
    if TAC_SEG_IMAGE_FORMAT == "png":
        return
    if not HAS_CV2:
        unreal.log_warning("  JPG convert: OpenCV unavailable; keeping PNG frames")
        return
    converted = 0
    for src_path in _iter_image_files(images_folder):
        base, ext = os.path.splitext(src_path)
        if ext.lower() in {".jpg", ".jpeg"}:
            continue
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        dst_path = f"{base}.jpg"
        if cv2.imwrite(dst_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), TAC_SEG_JPEG_QUALITY]):
            os.remove(src_path)
            converted += 1
    if converted:
        unreal.log(f"  Converted {converted} frames to JPG (quality={TAC_SEG_JPEG_QUALITY})")


def split_dataset(output_dir, val_ratio=0.2):
    staging_images = os.path.join(output_dir, "_staging", "images")
    staging_labels = os.path.join(output_dir, "_staging", "labels")
    all_images = _iter_image_files(staging_images)
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
            ext = os.path.splitext(img_path)[1].lower()
            shutil.move(img_path, os.path.join(img_dir, f"{base}{ext}"))
            lbl = os.path.join(staging_labels, f"{base}.txt")
            if os.path.exists(lbl):
                shutil.move(lbl, os.path.join(lbl_dir, f"{base}.txt"))
        unreal.log(f"  {split_name}: {len(img_list)} samples")
    staging = os.path.join(output_dir, "_staging")
    if os.path.exists(staging):
        shutil.rmtree(staging)


def generate_data_yaml(output_dir, class_map, class_name_map=None):
    yaml_path = os.path.join(output_dir, "data.yaml")
    nc = len(class_map)
    id_to_name = dict(class_name_map) if class_name_map else {v: k for k, v in class_map.items()}
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.rstrip('/').rstrip(chr(92))}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("task: segment\n")
        f.write(f"nc: {nc}\n")
        f.write("names:\n")
        for idx in sorted(id_to_name.keys()):
            f.write(f"  {idx}: {id_to_name[idx]}\n")
    unreal.log(f"  data.yaml: nc={nc}, classes={id_to_name}, task=segment")


# =============================================================================
# ENTRY POINT
# =============================================================================

if 'tac_seg_gen' in dir():
    del tac_seg_gen

if 'global_executor' in dir() and global_executor:
    global_executor = None

tac_seg_gen = TACSegGenerator()
