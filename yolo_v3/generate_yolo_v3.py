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
    │   ├── gate_searchrescue/   (class 0 = gate_searchrescue)
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
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

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

YOLO_V3_MIN_BBOX_WIDTH_PX = getattr(_config_mod, "YOLO_V3_MIN_BBOX_WIDTH_PX", 4)
YOLO_V3_MIN_BBOX_HEIGHT_PX = getattr(_config_mod, "YOLO_V3_MIN_BBOX_HEIGHT_PX", 8)
YOLO_V3_OCCLUSION_MODE = str(getattr(_config_mod, "YOLO_V3_OCCLUSION_MODE", "off")).lower()
YOLO_V3_OCCLUSION_OVERLAP_RATIO = float(getattr(_config_mod, "YOLO_V3_OCCLUSION_OVERLAP_RATIO", 0.25))
YOLO_V3_OCCLUSION_DEPTH_MARGIN_CM = float(getattr(_config_mod, "YOLO_V3_OCCLUSION_DEPTH_MARGIN_CM", 10.0))
if YOLO_V3_OCCLUSION_MODE not in {"off", "drop", "refine"}:
    YOLO_V3_OCCLUSION_MODE = "off"

# Internal constants
# Show-only mask (detect) needs only 1 capture, so full res is affordable.
# Segment mode also uses full resolution for accurate polygon vertices.
MASK_RESOLUTION_X = RESOLUTION_X
MASK_RESOLUTION_Y = RESOLUTION_Y
RT_ASSET_PATH = "/Game/Generated/YOLOV3MaskRT"
SEQUENCE_PATH = YOLO_V3_SEQUENCE_PREFIX + "Sequence"
BOX_EDGE_INDICES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
)

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
    cv_x, cv_y, cv_z = _world_to_camera_cv(world_pt, cam_transform)
    return _project_camera_point(cv_x, cv_y, cv_z, intrinsics)


def _world_to_camera_cv(world_pt, cam_transform):
    cam_inv = cam_transform.inverse()
    local = unreal.MathLibrary.transform_location(cam_inv, world_pt)
    return local.y, -local.z, local.x


def _project_camera_point(cv_x, cv_y, cv_z, intrinsics):
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


def _vec_add(a, b):
    return unreal.Vector(a.x + b.x, a.y + b.y, a.z + b.z)


def _vec_sub(a, b):
    return unreal.Vector(a.x - b.x, a.y - b.y, a.z - b.z)


def _rotate_vector(rot, vec):
    return unreal.MathLibrary.transform_direction(
        unreal.Transform(rotation=rot),
        vec,
    )


# =============================================================================
# CAMERA POSITION GENERATORS
# =============================================================================

def generate_vertical_hemisphere(center, min_dist, max_dist, phi_max=90.0):
    """Camera orbits ABOVE — bird's eye to side views (for cam_bottom).

    phi_max: maximum polar angle in degrees from vertical (default 90 = full
             hemisphere).  Lower values cut a band off the equator so the
             camera stays more top-down.
    """
    dist = random.uniform(min_dist, max_dist)
    theta = random.uniform(0, 2 * math.pi)
    # uniform sampling on the spherical cap [0, phi_max]
    cos_limit = math.cos(math.radians(phi_max))
    phi = math.acos(random.uniform(cos_limit, 1))  # uniform hemisphere cap
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
    return generate_vertical_hemisphere(center, min_d, max_d,
                                        phi_max=obj_config.get("phi_max", 90.0))


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


def _position_capture_actor(capture_actor, cine_camera, cam_pos, cam_rot):
    """Align SceneCapture2D to the final evaluated CineCamera transform."""
    cc = capture_actor.capture_component2d
    cine_camera.set_actor_location_and_rotation(cam_pos, cam_rot, False, True)
    try:
        cine_comp = cine_camera.get_cine_camera_component()
        actual_pos = cine_comp.get_world_location()
        actual_rot = cine_comp.get_world_rotation()
    except (AttributeError, Exception):
        actual_pos, actual_rot = cam_pos, cam_rot
    cc.set_world_location_and_rotation(actual_pos, actual_rot, False, True)
    return actual_pos, actual_rot


def _capture_scene_rgb(capture_actor, render_target, cine_camera, cam_pos, cam_rot):
    """Capture the current scene from the provided camera pose."""
    cc = capture_actor.capture_component2d
    _position_capture_actor(capture_actor, cine_camera, cam_pos, cam_rot)
    cc.capture_scene()
    return _read_render_target_as_numpy(render_target)


def _binary_mask_from_diff(fg_pixels, bg_pixels, threshold=25):
    """Extract a binary visibility mask from two RGB captures."""
    if fg_pixels is None or bg_pixels is None or not HAS_NUMPY:
        return None

    diff_gray = np.abs(fg_pixels.astype(np.int16) - bg_pixels.astype(np.int16)).max(axis=2)
    binary = (diff_gray >= threshold).astype(np.uint8) * 255

    if HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def _extract_bbox_from_mask(binary_mask):
    """Return normalized bbox from a binary mask, or None if empty."""
    if binary_mask is None or not HAS_NUMPY:
        return None
    ys, xs = np.nonzero(binary_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = float(xs.min())
    x2 = float(xs.max() + 1)
    y1 = float(ys.min())
    y2 = float(ys.max() + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    return (
        ((x1 + x2) / 2.0) / RESOLUTION_X,
        ((y1 + y2) / 2.0) / RESOLUTION_Y,
        (x2 - x1) / RESOLUTION_X,
        (y2 - y1) / RESOLUTION_Y,
    )


def capture_differential_mask(capture_actor, render_target, cine_camera,
                               cam_pos, cam_rot, target_actor):
    """Two-pass differential mask. Returns binary mask (HxW uint8) or None."""
    # Pass 1: background (target hidden)
    target_actor.set_actor_hidden_in_game(True)
    bg = _capture_scene_rgb(capture_actor, render_target, cine_camera, cam_pos, cam_rot)

    # Pass 2: foreground (target visible)
    target_actor.set_actor_hidden_in_game(False)
    fg = _capture_scene_rgb(capture_actor, render_target, cine_camera, cam_pos, cam_rot)
    return _binary_mask_from_diff(fg, bg)


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


def _get_annotation_world_boxes(actor):
    """Return annotation boxes as lists of 8 world-space corners.

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
            return [[unreal.MathLibrary.transform_location(comp_tf, c) for c in local_corners]]

    boxes = []
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
        boxes.append([
            unreal.MathLibrary.transform_location(comp_tf, c) for c in local_corners
        ])

    if boxes:
        return boxes

    origin, extent = actor.get_actor_bounds(False)
    ex, ey, ez = extent.x, extent.y, extent.z
    ox, oy, oz = origin.x, origin.y, origin.z
    return [[
        unreal.Vector(ox + sx * ex, oy + sy * ey, oz + sz * ez)
        for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)
    ]]


def _get_annotation_world_corners(actor):
    """Return flattened world-space corners for labeling/centers."""
    return [corner for box in _get_annotation_world_boxes(actor) for corner in box]


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


def _get_bottom_pivot_local_offset(actor, actor_loc=None, actor_rot=None):
    """Return the annotation box bottom-center in actor-local space."""
    if actor_loc is None:
        actor_loc = actor.get_actor_location()
    if actor_rot is None:
        actor_rot = actor.get_actor_rotation()
    actor_tf_inv = unreal.Transform(location=actor_loc, rotation=actor_rot).inverse()
    local_corners = [
        unreal.MathLibrary.transform_location(actor_tf_inv, corner)
        for corner in _get_annotation_world_corners(actor)
    ]
    xs = [c.x for c in local_corners]
    ys = [c.y for c in local_corners]
    zs = [c.z for c in local_corners]
    return unreal.Vector(
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
        min(zs),
    )


def _get_2d_bbox_obb(actor, cam_transform, intrinsics):
    """Tight 2D bounding box via clipped OBB edge projection.

    Raw corner clamping can create giant boxes when only a tiny tip is visible
    and the rest of the actor is off-screen. This clips each 3D box edge to
    the camera near plane and the image bounds before computing the final bbox.
    """
    pts = []
    for world_box in _get_annotation_world_boxes(actor):
        pts.extend(_project_clipped_box_points(world_box, cam_transform, intrinsics))

    if len(pts) < 2:
        return None

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        max(0.0, min(1.0, ((x1 + x2) / 2.0) / RESOLUTION_X)),
        max(0.0, min(1.0, ((y1 + y2) / 2.0) / RESOLUTION_Y)),
        max(0.0, min(1.0, (x2 - x1) / RESOLUTION_X)),
        max(0.0, min(1.0, (y2 - y1) / RESOLUTION_Y)),
    )


def _bbox_meets_min_size(bbox):
    """Filter out projected boxes that are too small to be useful."""
    if not bbox:
        return False
    _, _, w_norm, h_norm = bbox
    width_px = w_norm * RESOLUTION_X
    height_px = h_norm * RESOLUTION_Y
    return (
        width_px >= YOLO_V3_MIN_BBOX_WIDTH_PX and
        height_px >= YOLO_V3_MIN_BBOX_HEIGHT_PX
    )


def _bbox_touches_edge(bbox, margin_px=1.0):
    """Return True if the bbox touches or extends beyond any image edge."""
    if not bbox:
        return False
    xc, yc, w, h = bbox
    margin_x = margin_px / RESOLUTION_X
    margin_y = margin_px / RESOLUTION_Y
    return (
        (xc - w / 2.0) <= margin_x or
        (xc + w / 2.0) >= 1.0 - margin_x or
        (yc - h / 2.0) <= margin_y or
        (yc + h / 2.0) >= 1.0 - margin_y
    )


def _clip_segment_to_near_plane(p0, p1, near_z=1.0):
    """Clip a camera-space segment against the near plane z >= near_z."""
    z0, z1 = p0[2], p1[2]
    if z0 < near_z and z1 < near_z:
        return None
    if z0 >= near_z and z1 >= near_z:
        return p0, p1

    t = (near_z - z0) / (z1 - z0)
    clipped = (
        p0[0] + (p1[0] - p0[0]) * t,
        p0[1] + (p1[1] - p0[1]) * t,
        near_z,
    )
    if z0 < near_z:
        return clipped, p1
    return p0, clipped


def _clip_line_to_screen(p0, p1):
    """Clip a 2D line segment to the image rectangle."""
    x_min, y_min = 0.0, 0.0
    x_max, y_max = float(RESOLUTION_X), float(RESOLUTION_Y)

    def _out_code(x, y):
        code = 0
        if x < x_min:
            code |= 1
        elif x > x_max:
            code |= 2
        if y < y_min:
            code |= 4
        elif y > y_max:
            code |= 8
        return code

    x0, y0 = p0
    x1, y1 = p1
    while True:
        code0 = _out_code(x0, y0)
        code1 = _out_code(x1, y1)

        if not (code0 | code1):
            return (x0, y0), (x1, y1)
        if code0 & code1:
            return None

        out_code = code0 or code1
        if out_code & 8:
            x = x0 + (x1 - x0) * (y_max - y0) / (y1 - y0)
            y = y_max
        elif out_code & 4:
            x = x0 + (x1 - x0) * (y_min - y0) / (y1 - y0)
            y = y_min
        elif out_code & 2:
            y = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0)
            x = x_max
        else:
            y = y0 + (y1 - y0) * (x_min - x0) / (x1 - x0)
            x = x_min

        if out_code == code0:
            x0, y0 = x, y
        else:
            x1, y1 = x, y


def _project_clipped_box_points(world_box, cam_transform, intrinsics):
    """Project only the on-screen portion of a single 3D box."""
    visible_points = []
    cam_pts = [_world_to_camera_cv(corner, cam_transform) for corner in world_box]

    for cv_x, cv_y, cv_z in cam_pts:
        pt2d = _project_camera_point(cv_x, cv_y, cv_z, intrinsics)
        if pt2d != [-9999.0, -9999.0]:
            if 0.0 <= pt2d[0] <= RESOLUTION_X and 0.0 <= pt2d[1] <= RESOLUTION_Y:
                visible_points.append((pt2d[0], pt2d[1]))

    for idx0, idx1 in BOX_EDGE_INDICES:
        clipped_3d = _clip_segment_to_near_plane(cam_pts[idx0], cam_pts[idx1])
        if not clipped_3d:
            continue
        p0_2d = _project_camera_point(*clipped_3d[0], intrinsics)
        p1_2d = _project_camera_point(*clipped_3d[1], intrinsics)
        clipped_2d = _clip_line_to_screen(tuple(p0_2d), tuple(p1_2d))
        if clipped_2d:
            visible_points.extend(clipped_2d)

    deduped = []
    seen = set()
    for x, y in visible_points:
        key = (round(x, 4), round(y, 4))
        if key not in seen:
            seen.add(key)
            deduped.append((x, y))
    return deduped


def _bbox_to_pixel_rect(bbox):
    """Convert normalized bbox to pixel rect (x1, y1, x2, y2)."""
    xc, yc, w, h = bbox
    width_px = w * RESOLUTION_X
    height_px = h * RESOLUTION_Y
    return (
        (xc * RESOLUTION_X) - (width_px / 2.0),
        (yc * RESOLUTION_Y) - (height_px / 2.0),
        (xc * RESOLUTION_X) + (width_px / 2.0),
        (yc * RESOLUTION_Y) + (height_px / 2.0),
    )


def _rect_area(rect):
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _rect_intersection_area(rect_a, rect_b):
    x1 = max(rect_a[0], rect_b[0])
    y1 = max(rect_a[1], rect_b[1])
    x2 = min(rect_a[2], rect_b[2])
    y2 = min(rect_a[3], rect_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _cross_2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points):
    """Return convex hull in CCW order using the monotonic chain algorithm."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 1:
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


def _polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def _ensure_ccw(poly):
    if len(poly) < 3:
        return poly
    signed_area = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        signed_area += (x2 - x1) * (y2 + y1)
    return list(reversed(poly)) if signed_area > 0 else poly


def _line_intersection(p1, p2, q1, q2):
    """Return the intersection point between two infinite 2D lines."""
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]
    a2 = q2[1] - q1[1]
    b2 = q1[0] - q2[0]
    c2 = a2 * q1[0] + b2 * q1[1]
    det = (a1 * b2) - (a2 * b1)
    if abs(det) < 1e-6:
        return p2
    return (
        ((b2 * c1) - (b1 * c2)) / det,
        ((a1 * c2) - (a2 * c1)) / det,
    )


def _convex_polygon_intersection(subject, clip_poly):
    """Clip a convex polygon by another convex polygon (both CCW)."""
    output = list(subject)
    clip_poly = _ensure_ccw(list(clip_poly))
    if len(output) < 3 or len(clip_poly) < 3:
        return []

    for i in range(len(clip_poly)):
        cp1 = clip_poly[i]
        cp2 = clip_poly[(i + 1) % len(clip_poly)]
        input_poly = output
        output = []
        if not input_poly:
            break

        s = input_poly[-1]
        for e in input_poly:
            e_inside = _cross_2d(cp1, cp2, e) >= 0
            s_inside = _cross_2d(cp1, cp2, s) >= 0
            if e_inside:
                if not s_inside:
                    output.append(_line_intersection(s, e, cp1, cp2))
                output.append(e)
            elif s_inside:
                output.append(_line_intersection(s, e, cp1, cp2))
            s = e
    return output


def _projected_overlap_ratio(poly_a, poly_b):
    """Return overlap area / smaller polygon area for two projected convex shapes."""
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    area_a = _polygon_area(poly_a)
    area_b = _polygon_area(poly_b)
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0
    intersection = _convex_polygon_intersection(_ensure_ccw(poly_a), _ensure_ccw(poly_b))
    inter_area = _polygon_area(intersection)
    if inter_area <= 1e-6:
        return 0.0
    return inter_area / max(1.0, min(area_a, area_b))


def _get_projected_actor_shape(actor, cam_transform, intrinsics):
    """Return projected bbox/hull/depth for one actor in the current frame."""
    pts = []
    for world_box in _get_annotation_world_boxes(actor):
        pts.extend(_project_clipped_box_points(world_box, cam_transform, intrinsics))
    if len(pts) < 2:
        return None

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None

    bbox = (
        max(0.0, min(1.0, ((x1 + x2) / 2.0) / RESOLUTION_X)),
        max(0.0, min(1.0, ((y1 + y2) / 2.0) / RESOLUTION_Y)),
        max(0.0, min(1.0, (x2 - x1) / RESOLUTION_X)),
        max(0.0, min(1.0, (y2 - y1) / RESOLUTION_Y)),
    )

    center_depth = _world_to_camera_cv(_get_annotation_center(actor), cam_transform)[2]
    hull = _convex_hull(pts)
    return {
        "bbox": bbox,
        "bbox_rect_px": _bbox_to_pixel_rect(bbox),
        "hull": hull,
        "depth_cm": center_depth,
    }


def _find_occlusion_targets(projected_infos):
    """Return actors whose farther projected shape is significantly overlapped."""
    if YOLO_V3_OCCLUSION_MODE == "off":
        return set()

    flagged = set()
    for i in range(len(projected_infos)):
        info_a = projected_infos[i]
        for j in range(i + 1, len(projected_infos)):
            info_b = projected_infos[j]

            inter_area = _rect_intersection_area(info_a["bbox_rect_px"], info_b["bbox_rect_px"])
            if inter_area <= 0.0:
                continue

            smaller_rect_area = min(_rect_area(info_a["bbox_rect_px"]), _rect_area(info_b["bbox_rect_px"]))
            if smaller_rect_area <= 0.0:
                continue
            if (inter_area / smaller_rect_area) < max(0.05, YOLO_V3_OCCLUSION_OVERLAP_RATIO * 0.5):
                continue

            if info_a["depth_cm"] <= 0.0 or info_b["depth_cm"] <= 0.0:
                continue

            depth_delta = abs(info_a["depth_cm"] - info_b["depth_cm"])
            if depth_delta < YOLO_V3_OCCLUSION_DEPTH_MARGIN_CM:
                continue

            overlap_ratio = _projected_overlap_ratio(info_a["hull"], info_b["hull"])
            if overlap_ratio < YOLO_V3_OCCLUSION_OVERLAP_RATIO:
                continue

            farther = info_a if info_a["depth_cm"] > info_b["depth_cm"] else info_b
            flagged.add(farther["actor"])
    return flagged


def _capture_refined_bbox(capture_actor, render_target, cine_camera, cam_pos, cam_rot,
                          full_scene_rgb, target_actor):
    """Capture a visible-only bbox for one flagged object using a single hidden pass."""
    if full_scene_rgb is None:
        return None
    try:
        target_actor.set_actor_hidden_in_game(True)
        hidden_scene = _capture_scene_rgb(capture_actor, render_target, cine_camera, cam_pos, cam_rot)
    finally:
        target_actor.set_actor_hidden_in_game(False)
    binary_mask = _binary_mask_from_diff(full_scene_rgb, hidden_scene)
    return _extract_bbox_from_mask(binary_mask)


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
        self.current_class_name_map = {}  # {class_id: display_name} for data.yaml
        self.current_sub_actors = []      # list of actors sharing the main target's class_id

        unreal.log("=" * 60)
        unreal.log("UE5.7 YOLO DATASET GENERATOR V3")
        unreal.log("  Per-Object | Registry-Driven | Multi-Hemisphere")
        unreal.log("=" * 60)

        if YOLO_V3_MODE == "segment" and not HAS_CV2:
            unreal.log_error("Segmentation mode requires cv2!")
            return
        if YOLO_V3_MODE == "detect" and YOLO_V3_OCCLUSION_MODE == "refine" and not HAS_NUMPY:
            unreal.log_error("Detect-mode occlusion refinement requires numpy!")
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
        if YOLO_V3_MODE == "detect":
            unreal.log(f"Detect occlusion mode: {YOLO_V3_OCCLUSION_MODE}")
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
        for actor in self.negative_hide_actors:
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
        # Use class_name override when available so merged datasets share class names
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
            unreal.log(f"  Co-visible: {[n for n, _ in self.current_co_visible]} → class map: {self.current_class_map}")

        # Resolve sub_actors (same class_id as their parent, each gets its own bbox)
        # Collect from main target AND from co-visible entries
        self.current_sub_actors = []  # list of (class_id_key, actor) — class_id_key indexes current_class_map
        for sub_label in obj_config.get("sub_actors", []):
            sub_actor = self._find_actor_by_label(sub_label)
            if sub_actor:
                self.current_sub_actors.append((obj_name, sub_actor))
            else:
                unreal.log_warning(f"  Sub-actor '{sub_label}' not found in scene")
        for co_name, _ in self.current_co_visible:
            try:
                co_cfg = get_object_config(co_name)
                for sub_label in co_cfg.get("sub_actors", []):
                    sub_actor = self._find_actor_by_label(sub_label)
                    if sub_actor:
                        self.current_sub_actors.append((co_name, sub_actor))
                    else:
                        unreal.log_warning(f"  Sub-actor '{sub_label}' (from co-visible '{co_name}') not found")
            except KeyError:
                pass
        if self.current_sub_actors:
            unreal.log(f"  Sub-actors: {[(k, a.get_actor_label()) for k, a in self.current_sub_actors]}")

        # Resolve keep_visible labels: collect from target + co-visible configs + sub_actors
        self.current_keep_visible_labels = set(obj_config.get("keep_visible", []))
        for co_name, _ in self.current_co_visible:
            try:
                co_cfg = get_object_config(co_name)
                self.current_keep_visible_labels.update(co_cfg.get("keep_visible", []))
            except KeyError:
                pass
        if self.current_keep_visible_labels:
            unreal.log(f"  Keep-visible HideInNegative actors: {sorted(self.current_keep_visible_labels)}")

        # Resolve hard_negative_actors: HideInNegative actors shown ONLY in negative frames
        # (underground in positives). Teaches model these shapes are not the target class.
        self.current_hard_negative_labels = set(obj_config.get("hard_negative_actors", []))
        self.current_hard_negative_actors = []
        for hide_actor in self.negative_hide_actors:
            if hide_actor.get_actor_label() in self.current_hard_negative_labels:
                self.current_hard_negative_actors.append(hide_actor)
        if self.current_hard_negative_actors:
            unreal.log(f"  Hard-negative actors (visible only in negatives): "
                       f"{[a.get_actor_label() for a in self.current_hard_negative_actors]}")
        elif self.current_hard_negative_labels:
            unreal.log_warning(
                f"  hard_negative_actors requested {sorted(self.current_hard_negative_labels)} "
                f"but none matched HideInNegative actors in scene. "
                f"Available: {[a.get_actor_label() for a in self.negative_hide_actors]}")

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
        sub_actors = {a for _, a in self.current_sub_actors}
        keep_above = co_visible_actors | sub_actors
        self.non_target_original_locs = {}
        for actor in self.all_target_actors:
            if actor != target_actor and actor not in keep_above:
                self.non_target_original_locs[actor] = actor.get_actor_location()
                loc = actor.get_actor_location()
                actor.set_actor_location(unreal.Vector(loc.x, loc.y, -20000.0), False, False)
        # Also hide HideInNegative actors not listed in keep_visible or hard_negative_actors
        hidden_hide_count = 0
        keep_labels = self.current_keep_visible_labels | self.current_hard_negative_labels
        for actor in self.negative_hide_actors:
            if actor.get_actor_label() not in keep_labels:
                self.non_target_original_locs[actor] = actor.get_actor_location()
                loc = actor.get_actor_location()
                actor.set_actor_location(unreal.Vector(loc.x, loc.y, -20000.0), False, False)
                hidden_hide_count += 1
        if self.non_target_original_locs:
            unreal.log(f"  Hidden {len(self.non_target_original_locs)} non-target actor(s)"
                       f" ({hidden_hide_count} HideInNegative)")
        if keep_above:
            unreal.log(f"  Kept {len(keep_above)} co-visible/sub actor(s) above ground")

    def _restore_non_targets(self):
        for actor, orig_loc in self.non_target_original_locs.items():
            actor.set_actor_location(orig_loc, False, False)
        count = len(self.non_target_original_locs)
        self.non_target_original_locs = {}
        if count:
            unreal.log(f"  Restored {count} non-target actor(s)")

    def _resolve_rotation_dr(self, cfg, apply_to_self=False, apply_to_sub_actors=False):
        rotation_dr = cfg.get("rotation_dr")
        if not rotation_dr:
            return None
        if apply_to_self and not rotation_dr.get("apply_to_self", True):
            return None
        if apply_to_sub_actors and not rotation_dr.get("apply_to_sub_actors", False):
            return None
        return rotation_dr

    def _build_actor_track(self, actor, channels, rotation_dr=None):
        track = {
            "actor": actor,
            "channels": channels,
            "orig_loc": actor.get_actor_location(),
            "orig_rot": actor.get_actor_rotation(),
            "orig_scale": actor.get_actor_scale3d(),
        }
        if rotation_dr:
            mode = rotation_dr.get("mode", "bottom_pivot")
            if mode != "bottom_pivot":
                unreal.log_warning(
                    f"  Unsupported rotation_dr mode '{mode}' for '{actor.get_actor_label()}'")
            else:
                pivot_local = _get_bottom_pivot_local_offset(
                    actor, track["orig_loc"], track["orig_rot"])
                track["rotation_dr"] = {
                    "mode": mode,
                    "roll_range": rotation_dr.get("roll_range", 0.0),
                    "pitch_range": rotation_dr.get("pitch_range", 0.0),
                    "pivot_local": pivot_local,
                    "pivot_world": _vec_add(
                        track["orig_loc"],
                        _rotate_vector(track["orig_rot"], pivot_local),
                    ),
                }
        return track

    def _write_transform_keys(self, channels, frame_time, loc, rot, scale):
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
            loc = track["orig_loc"]
            rot = track["orig_rot"]
        self._write_transform_keys(track["channels"], frame_time, loc, rot, track["orig_scale"])

    def _sample_track_pose(self, track):
        rotation_dr = track.get("rotation_dr")
        if not rotation_dr:
            return track["orig_loc"], track["orig_rot"]

        rot = unreal.Rotator(
            roll=track["orig_rot"].roll + random.uniform(-rotation_dr["roll_range"], rotation_dr["roll_range"]),
            pitch=track["orig_rot"].pitch + random.uniform(-rotation_dr["pitch_range"], rotation_dr["pitch_range"]),
            yaw=track["orig_rot"].yaw,
        )
        rotated_pivot = _rotate_vector(rot, rotation_dr["pivot_local"])
        loc = _vec_sub(rotation_dr["pivot_world"], rotated_pivot)
        return loc, rot

    def _apply_actor_states(self, actor_states):
        for actor, state in actor_states.items():
            actor.set_actor_location_and_rotation(state["loc"], state["rot"], False, True)

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
        target_track_data = self._build_actor_track(
            target,
            target_channels,
            rotation_dr=self._resolve_rotation_dr(obj_config, apply_to_self=True),
        )
        orig_loc = target_track_data["orig_loc"]
        orig_rot = target_track_data["orig_rot"]
        orig_scale = target_track_data["orig_scale"]

        # Initial keyframe at frame 0
        frame_0 = unreal.FrameNumber(0)
        self._write_track_pose(target_track_data, frame_0)

        # Bind co-visible actors (keyframed: original pos for positive, underground for negative)
        co_visible_tracks = []
        for co_name, co_actor in self.current_co_visible:
            co_binding = seq.add_possessable(co_actor)
            co_track = co_binding.add_track(unreal.MovieScene3DTransformTrack)
            co_section = co_track.add_section()
            co_section.set_range(0, total_frames + 10)
            co_channels = co_section.get_all_channels()
            co_cfg = get_object_config(co_name)
            co_track_data = self._build_actor_track(
                co_actor,
                co_channels,
                rotation_dr=self._resolve_rotation_dr(co_cfg, apply_to_self=True),
            )
            self._write_track_pose(co_track_data, frame_0)
            co_visible_tracks.append(co_track_data)

        # Bind sub_actors (keyframed: original pos for positive, underground for negative)
        sub_actor_tracks = []
        for sub_key, sub_actor in self.current_sub_actors:
            sub_binding = seq.add_possessable(sub_actor)
            sub_track = sub_binding.add_track(unreal.MovieScene3DTransformTrack)
            sub_section = sub_track.add_section()
            sub_section.set_range(0, total_frames + 10)
            sub_channels = sub_section.get_all_channels()
            sub_cfg = obj_config if sub_key == obj_name else get_object_config(sub_key)
            sub_track_data = self._build_actor_track(
                sub_actor,
                sub_channels,
                rotation_dr=self._resolve_rotation_dr(sub_cfg, apply_to_sub_actors=True),
            )
            self._write_track_pose(sub_track_data, frame_0)
            sub_actor_tracks.append(sub_track_data)

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
            hide_track_data = self._build_actor_track(hide_actor, hide_channels)
            self._write_track_pose(hide_track_data, frame_0)
            negative_hide_tracks.append(hide_track_data)
        if negative_hide_tracks:
            unreal.log(f"  Negative-hide actors: {len(negative_hide_tracks)}")

        # Bind hard_negative actors (inverted: underground in positives, visible in negatives)
        hard_negative_tracks = []
        for hn_actor in self.current_hard_negative_actors:
            hn_binding = seq.add_possessable(hn_actor)
            hn_track = hn_binding.add_track(unreal.MovieScene3DTransformTrack)
            hn_section = hn_track.add_section()
            hn_section.set_range(0, total_frames + 10)
            hn_channels = hn_section.get_all_channels()
            hn_track_data = self._build_actor_track(hn_actor, hn_channels)
            self._write_track_pose(hn_track_data, frame_0)
            hard_negative_tracks.append(hn_track_data)
        if hard_negative_tracks:
            unreal.log(f"  Hard-negative actors: {len(hard_negative_tracks)}")

        # Build frame schedule
        num_positive = obj_config["samples"]
        num_negative = self.current_total_samples - num_positive
        frame_types = ['positive'] * num_positive + ['negative'] * num_negative
        random.shuffle(frame_types)

        placement = obj_config.get("placement")
        jitter_enabled = obj_config.get("enable_jitter", True)
        jitter_pitch = obj_config.get("jitter_max_pitch", 5.0)
        filter_edges = not obj_config.get("samples_on_edges", True)

        for i in range(self.current_total_samples):
            frame_num = i * frames_per_sample
            frame_time = unreal.FrameNumber(frame_num)
            is_negative = (frame_types[i] == 'negative')

            if is_negative:
                # Target and friends go underground
                self._write_track_pose(target_track_data, frame_time, underground=True)
                for co_data in co_visible_tracks:
                    self._write_track_pose(co_data, frame_time, underground=True)
                for sub_data in sub_actor_tracks:
                    self._write_track_pose(sub_data, frame_time, underground=True)
                for hide_data in negative_hide_tracks:
                    self._write_track_pose(hide_data, frame_time, underground=True)

                if hard_negative_tracks:
                    # Hard-negative negatives: orbit the hard-negative actor using
                    # the target's hemisphere/distance config so the model sees the
                    # confusing shape at the same angles/distances as the real target.
                    hn_actor = self.current_hard_negative_actors[
                        i % len(self.current_hard_negative_actors)]
                    hn_center = _get_annotation_center(hn_actor)
                    cam_pos = generate_camera_position(hn_center, obj_config)
                    cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, hn_center)

                    # Apply jitter (same as positive frames)
                    if jitter_enabled:
                        dist = math.sqrt((cam_pos.x - hn_center.x)**2 +
                                         (cam_pos.y - hn_center.y)**2 +
                                         (cam_pos.z - hn_center.z)**2)
                        max_offset = dist * math.tan(math.radians(jitter_pitch))
                        jitter_scale = 1.0
                        for _ in range(4):
                            off = max_offset * jitter_scale
                            look_pt = unreal.Vector(
                                hn_center.x + random.uniform(-off, off),
                                hn_center.y + random.uniform(-off, off),
                                hn_center.z + random.uniform(-off * 0.5, off * 0.5))
                            cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, look_pt)
                            break

                    for hn_data in hard_negative_tracks:
                        self._write_track_pose(hn_data, frame_time)
                else:
                    # Standard negatives: random camera looking at pool floor
                    cam_pos = unreal.Vector(
                        random.uniform(POOL_BOUNDS["x_min"], POOL_BOUNDS["x_max"]),
                        random.uniform(POOL_BOUNDS["y_min"], POOL_BOUNDS["y_max"]),
                        random.uniform(POOL_BOUNDS["z_min"], POOL_BOUNDS["z_max"]))
                    cam_rot = unreal.Rotator(
                        roll=0.0,
                        pitch=random.uniform(-70.0, 0.0),
                        yaw=random.uniform(0.0, 360.0))

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
                    target_loc, target_rot = self._sample_track_pose(target_track_data)

                self._write_track_pose(target_track_data, frame_time, target_loc, target_rot)

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

                # Keyframe co-visible and sub actors for positive frames
                frame_actor_states = {
                    target: {"loc": target_loc, "rot": target_rot},
                }
                edge_hidden = set()

                # Sample poses, then optionally filter edge-touching actors
                secondary_tracks = (
                    [(d, *self._sample_track_pose(d)) for d in co_visible_tracks] +
                    [(d, *self._sample_track_pose(d)) for d in sub_actor_tracks]
                )
                if filter_edges and secondary_tracks:
                    cam_tf = unreal.Transform(location=cam_pos, rotation=cam_rot)
                    for track_data, loc, rot in secondary_tracks:
                        actor = track_data["actor"]
                        actor.set_actor_location_and_rotation(loc, rot, False, True)
                        bbox = _get_2d_bbox_obb(actor, cam_tf, self.intrinsics)
                        if bbox and _bbox_touches_edge(bbox):
                            edge_hidden.add(actor)

                for track_data, loc, rot in secondary_tracks:
                    if track_data["actor"] in edge_hidden:
                        self._write_track_pose(track_data, frame_time, underground=True)
                    else:
                        self._write_track_pose(track_data, frame_time, loc, rot)
                        frame_actor_states[track_data["actor"]] = {"loc": loc, "rot": rot}

                for hide_data in negative_hide_tracks:
                    self._write_track_pose(hide_data, frame_time)

                # Hard-negative actors: UNDERGROUND in positives (no false negatives)
                for hn_data in hard_negative_tracks:
                    self._write_track_pose(hn_data, frame_time, underground=True)

                self.current_sample_data.append({
                    "frame_idx": i, "target": target,
                    "cam_pos": cam_pos, "cam_rot": cam_rot, "is_negative": False,
                    "actor_states": frame_actor_states,
                    "edge_hidden": edge_hidden,
                })

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
        for sub_data in sub_actor_tracks:
            for ch in sub_data['channels']:
                for key in ch.get_keys():
                    key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)
        for hide_data in negative_hide_tracks:
            for ch in hide_data['channels']:
                for key in ch.get_keys():
                    key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)
        for hn_data in hard_negative_tracks:
            for ch in hn_data['channels']:
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
        if not obj_config.get("samples_on_edges", True):
            edge_hidden_count = sum(
                1 for d in self.current_sample_data
                if not d["is_negative"] and d.get("edge_hidden")
            )
            unreal.log(f"  Edge filtering: {edge_hidden_count}/{num_positive} positive frames had actors hidden")
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
        """Detect mode: geometric projection with optional targeted occlusion refinement."""
        unreal.log(f"  Generating labels (detect mode, occlusion={YOLO_V3_OCCLUSION_MODE})...")

        total_annotations = 0
        empty_frames = 0
        flagged_frames = 0
        refined_boxes = 0
        dropped_occluded = 0

        effective_occlusion_mode = YOLO_V3_OCCLUSION_MODE
        capture_actor = None
        render_target = None
        if effective_occlusion_mode == "refine":
            capture_actor, render_target = setup_scene_capture(self.camera)
            if not capture_actor:
                unreal.log_warning("  SceneCapture failed; falling back to pure geometric detect labels.")
                effective_occlusion_mode = "off"

        try:
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
                    self._apply_actor_states(data.get("actor_states", {}))
                    self.camera.set_actor_location_and_rotation(cam_pos, cam_rot, False, True)
                    try:
                        cine_comp = self.camera.get_cine_camera_component()
                        actual_pos = cine_comp.get_world_location()
                        actual_rot = cine_comp.get_world_rotation()
                    except (AttributeError, Exception):
                        actual_pos, actual_rot = cam_pos, cam_rot
                    cam_tf = unreal.Transform(location=actual_pos, rotation=actual_rot)
                    label_lines = []

                    edge_hidden = data.get("edge_hidden", set())
                    actors_to_label = []
                    if not obj_config.get("skip_target_bbox", False):
                        actors_to_label.append((self.current_class_map[obj_name], target, obj_name))
                    for co_name, co_actor in self.current_co_visible:
                        if co_actor not in edge_hidden:
                            actors_to_label.append((self.current_class_map[co_name], co_actor, co_name))
                    for sub_key, sub_actor in self.current_sub_actors:
                        if sub_actor not in edge_hidden:
                            actors_to_label.append((self.current_class_map[sub_key], sub_actor, sub_key))

                    projected_infos = []
                    for class_id, label_actor, label_name in actors_to_label:
                        shape = _get_projected_actor_shape(label_actor, cam_tf, self.intrinsics)
                        if not shape:
                            continue
                        shape["class_id"] = class_id
                        shape["actor"] = label_actor
                        shape["label_name"] = label_name
                        projected_infos.append(shape)

                    flagged_actors = (
                        _find_occlusion_targets(projected_infos)
                        if effective_occlusion_mode != "off" else set()
                    )
                    if flagged_actors:
                        flagged_frames += 1

                    full_scene_rgb = None
                    if flagged_actors and effective_occlusion_mode == "refine":
                        full_scene_rgb = _capture_scene_rgb(
                            capture_actor, render_target, self.camera, cam_pos, cam_rot
                        )

                    for info in projected_infos:
                        bbox = info["bbox"]
                        if info["actor"] in flagged_actors:
                            if effective_occlusion_mode == "drop":
                                dropped_occluded += 1
                                continue
                            if effective_occlusion_mode == "refine" and full_scene_rgb is not None:
                                if not _bbox_meets_min_size(bbox):
                                    dropped_occluded += 1
                                    continue
                                refined_bbox = _capture_refined_bbox(
                                    capture_actor, render_target, self.camera,
                                    cam_pos, cam_rot, full_scene_rgb, info["actor"]
                                )
                                if _bbox_meets_min_size(refined_bbox):
                                    bbox = refined_bbox
                                    refined_boxes += 1
                                else:
                                    dropped_occluded += 1
                                    continue

                        if _bbox_meets_min_size(bbox):
                            xc, yc, w, h = bbox
                            label_lines.append(f"{info['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                            total_annotations += 1

                    with open(label_path, 'w') as f:
                        f.write("\n".join(label_lines) + ("\n" if label_lines else ""))
                    if not label_lines:
                        empty_frames += 1
                    if (i + 1) % 50 == 0:
                        unreal.log(f"    Progress: {i + 1}/{self.current_total_samples}")
        finally:
            if capture_actor:
                capture_actor.destroy_actor()
            if unreal.EditorAssetLibrary.does_asset_exist(RT_ASSET_PATH):
                unreal.EditorAssetLibrary.delete_asset(RT_ASSET_PATH)

        unreal.log(f"  Labels: {total_annotations} annotations, "
                   f"{empty_frames} empty frames out of {self.current_total_samples}")
        if effective_occlusion_mode != "off":
            unreal.log(f"  Occlusion: {flagged_frames} flagged frames, "
                       f"{refined_boxes} refined boxes, {dropped_occluded} dropped occluded boxes")

    def _generate_labels_detect_legacy(self, target, obj_config, obj_name):
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
                self._apply_actor_states(data.get("actor_states", {}))
                self.camera.set_actor_location_and_rotation(cam_pos, cam_rot, False, True)
                try:
                    cine_comp = self.camera.get_cine_camera_component()
                    actual_pos = cine_comp.get_world_location()
                    actual_rot = cine_comp.get_world_rotation()
                except (AttributeError, Exception):
                    actual_pos, actual_rot = cam_pos, cam_rot
                cam_tf = unreal.Transform(location=actual_pos, rotation=actual_rot)
                label_lines = []

                edge_hidden = data.get("edge_hidden", set())
                actors_to_label = []
                if not obj_config.get("skip_target_bbox", False):
                    actors_to_label.append((self.current_class_map[obj_name], target, obj_name))
                for co_name, co_actor in self.current_co_visible:
                    if co_actor not in edge_hidden:
                        actors_to_label.append((self.current_class_map[co_name], co_actor, co_name))
                for sub_key, sub_actor in self.current_sub_actors:
                    if sub_actor not in edge_hidden:
                        actors_to_label.append((self.current_class_map[sub_key], sub_actor, sub_key))

                for class_id, label_actor, label_name in actors_to_label:
                    bbox = _get_2d_bbox_obb(label_actor, cam_tf, self.intrinsics)
                    if _bbox_meets_min_size(bbox):
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
                self._apply_actor_states(data.get("actor_states", {}))
                self.camera.set_actor_location_and_rotation(cam_pos, cam_rot, False, True)
                label_lines = []

                edge_hidden = data.get("edge_hidden", set())
                actors_to_label = []
                if not obj_config.get("skip_target_bbox", False):
                    actors_to_label.append((self.current_class_map[obj_name], target, obj_name))
                for co_name, co_actor in self.current_co_visible:
                    if co_actor not in edge_hidden:
                        actors_to_label.append((self.current_class_map[co_name], co_actor, co_name))
                for sub_key, sub_actor in self.current_sub_actors:
                    if sub_actor not in edge_hidden:
                        actors_to_label.append((self.current_class_map[sub_key], sub_actor, sub_key))

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
        class_name_map = dict(self.current_class_name_map)
        generator = self

        global_executor = unreal.MoviePipelinePIEExecutor()

        def on_finished(executor, success):
            global global_executor
            unreal.log("=" * 60)
            unreal.log(f"RENDER COMPLETE: '{class_name}' — Success: {success}")
            flatten_and_renumber_frames(output_dir)
            split_dataset(output_dir, val_split)
            generate_data_yaml(output_dir, class_map, YOLO_V3_MODE, class_name_map)
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


def generate_data_yaml(output_dir, class_map, mode="detect", class_name_map=None):
    """Generate data.yaml. class_map is {name: id} dict (supports multi-class via co_visible).

    class_name_map: optional {id: display_name} override. When provided, uses these
    names instead of the class_map keys. This allows registry entries like "slalom"
    to produce class names like "red_pipe" that match standalone datasets for merging.
    """
    yaml_path = os.path.join(output_dir, "data.yaml")
    task_str = "segment" if mode == "segment" else "detect"
    nc = len(class_map)
    # Invert: {id: name} — use class_name_map override when available
    if class_name_map:
        id_to_name = dict(class_name_map)
    else:
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
