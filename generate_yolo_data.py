"""
UE5.7.1 YOLO Detection Dataset Generator - V2 (Visibility-Aware + Camera Jitter)
Generates synthetic training data for YOLO object detection

Output Format (per image):
    labels/{frame:06d}.txt - One line per object: class_id x_center y_center width height
    images/{frame:06d}.png - Rendered image
    classes.txt - Class name mapping

All coordinates are normalized to [0, 1] relative to image dimensions.
Reference: https://docs.ultralytics.com/datasets/detect/

Updates in V2:
- Replaced 3D AABB mathematical bounding boxes with SceneCapture two-pass 
  differential masking for pixel-perfect, occlusion-aware tight bounding boxes.
- Added Negative Sampling to reduce false positives.
- Added Camera Jitter to prevent center-bias.
- Added IGNORE_TAG cleanup.
- Re-architected sequence generation for robustness and speed.

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

# Try to import cv2 for bounding box extraction
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# =============================================================================
# CONFIGURATION (imported from config.py)
# =============================================================================
import sys
if '__file__' in dir():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    _script_dir = next((p for p in [os.getcwd()] + sys.path if os.path.isfile(os.path.join(p, 'config.py'))), '')
if _script_dir and _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from config import (
    TARGET_TAG, CAMERA_TAG, IGNORE_TAG,
    POOL_BOUNDS, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM, FOCAL_LENGTH_MM,
    RESOLUTION_X, RESOLUTION_Y, WARMUP_FRAMES, SPATIAL_SAMPLES, TEMPORAL_SAMPLES,
    YOLO_OUTPUT_FOLDER, YOLO_SEQUENCE_PATH, YOLO_SAMPLES_PER_OBJECT,
    YOLO_VAL_SPLIT_RATIO, YOLO_NEGATIVE_SAMPLE_RATIO,
    YOLO_ENABLE_CAM_JITTER, YOLO_CAM_JITTER_MAX_PITCH, YOLO_CAM_JITTER_MAX_YAW,
    YOLO_MIN_DISTANCE, YOLO_MAX_DISTANCE,
    YOLO_USE_MASK_BBOX, YOLO_MIN_CONTOUR_AREA, YOLO_MIN_VISIBLE_PIXELS,
    YOLO_RANDOMIZE_OBJECTS,
    YOLO_OBJECT_XY_RANGE_X, YOLO_OBJECT_XY_RANGE_Y,
    YOLO_OBJECT_YAW_RANGE, YOLO_OBJECT_ROLL_RANGE,
    YOLO_OBJECT_PITCH_MIN, YOLO_OBJECT_PITCH_MAX,
    YOLO_OBJECT_MIN_SEPARATION,
)

# Alias prefixed names to local names used throughout the script
OUTPUT_FOLDER = YOLO_OUTPUT_FOLDER
SEQUENCE_PATH = YOLO_SEQUENCE_PATH
SAMPLES_PER_OBJECT = YOLO_SAMPLES_PER_OBJECT
VAL_SPLIT_RATIO = YOLO_VAL_SPLIT_RATIO
NEGATIVE_SAMPLE_RATIO = YOLO_NEGATIVE_SAMPLE_RATIO
ENABLE_CAM_JITTER = YOLO_ENABLE_CAM_JITTER
CAM_JITTER_MAX_PITCH = YOLO_CAM_JITTER_MAX_PITCH
CAM_JITTER_MAX_YAW = YOLO_CAM_JITTER_MAX_YAW
MIN_DISTANCE = YOLO_MIN_DISTANCE
MAX_DISTANCE = YOLO_MAX_DISTANCE
USE_MASK_BBOX = YOLO_USE_MASK_BBOX
MIN_CONTOUR_AREA = YOLO_MIN_CONTOUR_AREA
MIN_VISIBLE_PIXELS = YOLO_MIN_VISIBLE_PIXELS
RANDOMIZE_OBJECTS = YOLO_RANDOMIZE_OBJECTS
OBJECT_XY_RANGE_X = YOLO_OBJECT_XY_RANGE_X
OBJECT_XY_RANGE_Y = YOLO_OBJECT_XY_RANGE_Y
OBJECT_YAW_RANGE = YOLO_OBJECT_YAW_RANGE
OBJECT_ROLL_RANGE = YOLO_OBJECT_ROLL_RANGE
OBJECT_PITCH_MIN = YOLO_OBJECT_PITCH_MIN
OBJECT_PITCH_MAX = YOLO_OBJECT_PITCH_MAX
OBJECT_MIN_SEPARATION = YOLO_OBJECT_MIN_SEPARATION

# Internal constants (not user-configurable)
MASK_RESOLUTION_X = RESOLUTION_X
MASK_RESOLUTION_Y = RESOLUTION_Y
RT_ASSET_PATH = "/Game/Generated/YOLOMaskRT"
MAX_COLLISION_RETRIES = 20

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
    fx = FOCAL_LENGTH_MM * px_per_mm_x
    fy = FOCAL_LENGTH_MM * px_per_mm_y
    cx = RESOLUTION_X / 2.0
    cy = RESOLUTION_Y / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def project_point(world_pt, cam_transform, intrinsics):
    cam_inv = cam_transform.inverse()
    local = unreal.MathLibrary.transform_location(cam_inv, world_pt)

    cv_x = local.y
    cv_y = -local.z
    cv_z = local.x

    if cv_z <= 0:
        return [-9999.0, -9999.0]

    u = (cv_x * intrinsics["fx"] / cv_z) + intrinsics["cx"]
    v = (cv_y * intrinsics["fy"] / cv_z) + intrinsics["cy"]
    return [u, v]


def generate_clamped_position(center):
    dist = random.uniform(MIN_DISTANCE, MAX_DISTANCE)
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.pi / 3.5, math.pi / 1.8)

    dx = dist * math.sin(phi) * math.cos(theta)
    dy = dist * math.sin(phi) * math.sin(theta)
    dz = dist * math.cos(phi)

    return unreal.Vector(
        max(POOL_BOUNDS["x_min"], min(center.x + dx, POOL_BOUNDS["x_max"])),
        max(POOL_BOUNDS["y_min"], min(center.y + dy, POOL_BOUNDS["y_max"])),
        max(POOL_BOUNDS["z_min"], min(center.z + dz, POOL_BOUNDS["z_max"]))
    )


# =============================================================================
# SCENE CAPTURE MASK EXTRACTION
# =============================================================================

def create_render_target_asset(width, height):
    if unreal.EditorAssetLibrary.does_asset_exist(RT_ASSET_PATH):
        unreal.EditorAssetLibrary.delete_asset(RT_ASSET_PATH)

    world = get_world()

    try:
        rt = unreal.RenderingLibrary.create_render_target2d(
            world, width=width, height=height,
            format=unreal.TextureRenderTargetFormat.RTF_RGBA8,
            clear_color=unreal.LinearColor(0, 0, 0, 1)
        )
        if rt:
            unreal.log(f"  Render target created via RenderingLibrary: {width}x{height} RGBA8")
            return rt
    except (AttributeError, Exception) as e:
        unreal.log_warning(f"  RenderingLibrary.create_render_target2d failed: {e}")

    try:
        pkg_path, asset_name = RT_ASSET_PATH.rsplit('/', 1)
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
        factory = unreal.CanvasRenderTarget2DFactoryNew()
        rt = asset_tools.create_asset(
            asset_name, pkg_path, unreal.CanvasRenderTarget2D, factory
        )
        if rt:
            rt.set_editor_property('size_x', width)
            rt.set_editor_property('size_y', height)
            rt.set_editor_property('render_target_format', unreal.TextureRenderTargetFormat.RTF_RGBA8)
            rt.set_editor_property('clear_color', unreal.LinearColor(0, 0, 0, 1))
            unreal.log(f"  Render target created via factory fallback: {width}x{height} RGBA8")
            return rt
    except (AttributeError, Exception) as e:
        unreal.log_warning(f"  Factory fallback failed: {e}")

    unreal.log_error("Failed to create render target!")
    return None


def setup_scene_capture(camera_actor):
    rt = create_render_target_asset(MASK_RESOLUTION_X, MASK_RESOLUTION_Y)
    if not rt:
        return None, None

    try:
        cine_comp = camera_actor.get_cine_camera_component()
        fov_degrees = cine_comp.field_of_view
    except (AttributeError, Exception):
        fov_degrees = 2.0 * math.atan(SENSOR_WIDTH_MM / (2.0 * FOCAL_LENGTH_MM))
        fov_degrees = fov_degrees * (180.0 / math.pi)

    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    capture_actor = subsys.spawn_actor_from_class(
        unreal.SceneCapture2D, unreal.Vector(), unreal.Rotator()
    )

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

    w = render_target.size_x
    h = render_target.size_y
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, c in enumerate(colors):
        y = idx // w
        x = idx % w
        pixels[y, x] = [c.b, c.g, c.r]
    return pixels


def _get_2d_bbox_fallback(actor, cam_transform, intrinsics):
    """Fast mathematical AABB bounding box via 3D corner projection.
    
    Handles edge-of-screen objects by measuring how much the clamped box
    inflates compared to the true projected area. If inflation exceeds
    MAX_BBOX_INFLATION, the object is skipped (it's mostly off-screen
    and the box would be unreliable).
    """
    MAX_BBOX_INFLATION = 1.5  # Skip if clamped box is >150% of projected box area

    origin, extent = actor.get_actor_bounds(False)
    ex, ey, ez = extent.x, extent.y, extent.z
    ox, oy, oz = origin.x, origin.y, origin.z

    corners_3d = [
        unreal.Vector(ox + ex, oy + ey, oz + ez),
        unreal.Vector(ox + ex, oy + ey, oz - ez),
        unreal.Vector(ox + ex, oy - ey, oz + ez),
        unreal.Vector(ox + ex, oy - ey, oz - ez),
        unreal.Vector(ox - ex, oy + ey, oz + ez),
        unreal.Vector(ox - ex, oy + ey, oz - ez),
        unreal.Vector(ox - ex, oy - ey, oz + ez),
        unreal.Vector(ox - ex, oy - ey, oz - ez),
    ]

    points_2d = []
    behind_count = 0
    for corner in corners_3d:
        pt = project_point(corner, cam_transform, intrinsics)
        if pt == [-9999.0, -9999.0]:
            behind_count += 1
        else:
            points_2d.append(pt)

    # Need at least 4 visible corners for a reasonable box
    if len(points_2d) < 4:
        return None

    x_coords = [p[0] for p in points_2d]
    y_coords = [p[1] for p in points_2d]

    # Raw (unclamped) projected bbox
    raw_x_min = min(x_coords)
    raw_x_max = max(x_coords)
    raw_y_min = min(y_coords)
    raw_y_max = max(y_coords)

    raw_w = raw_x_max - raw_x_min
    raw_h = raw_y_max - raw_y_min
    raw_area = max(raw_w * raw_h, 1.0)

    # Clamped to image bounds
    x_min = max(0, raw_x_min)
    x_max = min(RESOLUTION_X, raw_x_max)
    y_min = max(0, raw_y_min)
    y_max = min(RESOLUTION_Y, raw_y_max)

    if x_max <= x_min or y_max <= y_min:
        return None

    clamped_w = x_max - x_min
    clamped_h = y_max - y_min
    clamped_area = clamped_w * clamped_h

    # Check inflation: if the clamped box is much larger relative to
    # the true projected footprint, the object is mostly off-screen
    # and the bbox would be misleadingly large
    if clamped_area / raw_area > MAX_BBOX_INFLATION:
        return None

    x_center = ((x_min + x_max) / 2.0) / RESOLUTION_X
    y_center = ((y_min + y_max) / 2.0) / RESOLUTION_Y
    width = clamped_w / RESOLUTION_X
    height = clamped_h / RESOLUTION_Y

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return (x_center, y_center, width, height)


def generate_clamped_position(center):
    """Generate random camera position on an upper hemisphere above the target.

    Uses uniform hemisphere sampling (phi = acos(random)) so that the camera
    positions are evenly distributed across the hemisphere surface, covering
    everything from bird's-eye (phi≈0) to side views (phi≈π/2).
    """
    dist = random.uniform(MIN_DISTANCE, MAX_DISTANCE)
    theta = random.uniform(0, 2 * math.pi)
    # Uniform hemisphere sampling: acos(U) where U ~ Uniform(0, 1)
    # phi=0 → directly above (bird's eye), phi=π/2 → horizontal (side view)
    phi = math.acos(random.uniform(0, 1))

    dx = dist * math.sin(phi) * math.cos(theta)
    dy = dist * math.sin(phi) * math.sin(theta)
    dz = dist * math.cos(phi)

    return unreal.Vector(
        max(POOL_BOUNDS["x_min"], min(center.x + dx, POOL_BOUNDS["x_max"])),
        max(POOL_BOUNDS["y_min"], min(center.y + dy, POOL_BOUNDS["y_max"])),
        max(POOL_BOUNDS["z_min"], min(center.z + dz, POOL_BOUNDS["z_max"]))
    )


def _check_no_overlap_2d(targets, transforms, min_sep):
    """Return True if no pair of objects overlaps in the XY plane.

    Uses each actor's bounding-box max(extent.x, extent.y) as a radius,
    then checks every pair for dist_2d < r_a + r_b + min_sep.
    """
    entries = []
    for t in targets:
        loc = transforms[t]['loc']
        _, extent = t.get_actor_bounds(False)  # half-extents
        radius = max(extent.x, extent.y)
        entries.append((loc.x, loc.y, radius))

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            ax, ay, ra = entries[i]
            bx, by, rb = entries[j]
            dist_2d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            if dist_2d < (ra + rb + min_sep):
                return False
    return True


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class YOLODatasetGenerator:
    def __init__(self):
        self.targets = []
        self.camera = None
        self.intrinsics = calculate_intrinsics()
        self.sample_data = []
        self.class_map = {}
        self.capture_actor = None
        self.render_target = None
        self.total_samples = 0

        unreal.log("=" * 60)
        unreal.log("UE5.7 YOLO DETECTION DATASET GENERATOR V2")
        unreal.log("  (Occlusion-Aware tight bounding boxes + Camera Jitter)")
        unreal.log("=" * 60)

        if not HAS_CV2:
            unreal.log_warning(
                "WARNING: cv2 not found - using old AABB mathematical bounding boxes.\n"
                "  To enable high-quality mask-based bounding boxes, install opencv-python-headless."
            )

        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
        self.staging_images = os.path.join(OUTPUT_FOLDER, "_staging", "images")
        self.staging_labels = os.path.join(OUTPUT_FOLDER, "_staging", "labels")
        os.makedirs(self.staging_images)
        os.makedirs(self.staging_labels)

        self._find_actors()
        if not self.camera or not self.targets:
            unreal.log_error("ERROR: Camera or targets not found!")
            return

        self._build_class_map()

        unreal.log(f"Found {len(self.targets)} targets, {len(self.class_map)} classes")
        unreal.log(f"Camera: {self.camera.get_actor_label()}")

        num_positive = SAMPLES_PER_OBJECT * len(self.targets)
        num_negative = int(num_positive * NEGATIVE_SAMPLE_RATIO / (1 - NEGATIVE_SAMPLE_RATIO))
        self.total_samples = num_positive + num_negative

        unreal.log(f"Samples: {SAMPLES_PER_OBJECT}/object * {len(self.targets)} = {num_positive} pos "
                   f"+ {num_negative} neg = {self.total_samples} total frames")

        self._configure_camera()
        self._run()

    def _configure_camera(self):
        if not self.camera:
            return
        cine_comp = self.camera.get_cine_camera_component()
        cine_comp.filmback.sensor_width = SENSOR_WIDTH_MM
        cine_comp.filmback.sensor_height = SENSOR_HEIGHT_MM
        cine_comp.current_focal_length = FOCAL_LENGTH_MM
        cine_comp.focus_settings.focus_method = unreal.CameraFocusMethod.DISABLE

    def _find_actors(self):
        subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        stale_count = 0
        ignored_count = 0
        
        for actor in subsys.get_all_level_actors():
            if actor.actor_has_tag(TARGET_TAG):
                self.targets.append(actor)
            elif actor.actor_has_tag(CAMERA_TAG):
                self.camera = actor
            elif actor.actor_has_tag(IGNORE_TAG):
                actor.destroy_actor()
                ignored_count += 1
            elif actor.get_class().get_name() == "SceneCapture2D":
                actor.destroy_actor()
                stale_count += 1
                
        if stale_count:
            unreal.log(f"  Cleaned up {stale_count} stale SceneCapture2D actor(s)")
        if ignored_count:
            unreal.log(f"  Removed {ignored_count} ignored object(s) tagged '{IGNORE_TAG}'")

    def _build_class_map(self):
        labels = sorted(set(t.get_actor_label() for t in self.targets))
        self.class_map = {label: idx for idx, label in enumerate(labels)}
        unreal.log(f"Class map: {self.class_map}")

    def _run(self):
        sequence = self._create_sequence()
        if not sequence:
            return

        self._generate_all_labels()
        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
        self._render(sequence)

    def _create_sequence(self):
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

        if unreal.EditorAssetLibrary.does_asset_exist(SEQUENCE_PATH):
            unreal.EditorAssetLibrary.delete_asset(SEQUENCE_PATH)

        pkg_path, asset_name = SEQUENCE_PATH.rsplit('/', 1)
        seq = asset_tools.create_asset(
            asset_name, pkg_path, unreal.LevelSequence, unreal.LevelSequenceFactoryNew()
        )

        seq.set_display_rate(unreal.FrameRate(24, 1))

        cam_binding = seq.add_possessable(self.camera)
        transform_track = cam_binding.add_track(unreal.MovieScene3DTransformTrack)
        transform_section = transform_track.add_section()

        frames_per_sample = 2
        total_frames = self.total_samples * frames_per_sample

        transform_section.set_range(0, total_frames + 10)
        seq.set_playback_start(0)
        seq.set_playback_end(total_frames)

        channels = transform_section.get_all_channels()

        target_tracks = {}
        for target in self.targets:
            binding = seq.add_possessable(target)
            track = binding.add_track(unreal.MovieScene3DTransformTrack)
            section = track.add_section()
            section.set_range(0, total_frames + 10)
            target_tracks[target] = {
                'channels': section.get_all_channels(),
                'orig_loc': target.get_actor_location(),
                'orig_rot': target.get_actor_rotation()
            }

        frame_0 = unreal.FrameNumber(0)
        for t, t_data in target_tracks.items():
            c = t_data['channels']
            o_loc, o_rot = t_data['orig_loc'], t_data['orig_rot']
            c[0].add_key(frame_0, o_loc.x)
            c[1].add_key(frame_0, o_loc.y)
            c[2].add_key(frame_0, o_loc.z)
            c[3].add_key(frame_0, o_rot.roll)
            c[4].add_key(frame_0, o_rot.pitch)
            c[5].add_key(frame_0, o_rot.yaw)

        num_positive = SAMPLES_PER_OBJECT * len(self.targets)
        num_negative = self.total_samples - num_positive

        target_assignments = []
        for target in self.targets:
            target_assignments.extend([target] * SAMPLES_PER_OBJECT)
        random.shuffle(target_assignments)

        frame_types = ['positive'] * num_positive + ['negative'] * num_negative
        random.shuffle(frame_types)

        positive_idx = 0
        last_is_negative = False
        
        for i in range(self.total_samples):
            frame_num = i * frames_per_sample
            frame_time = unreal.FrameNumber(frame_num)

            is_negative = (frame_types[i] == 'negative')

            if is_negative:
                cam_pos = unreal.Vector(
                    random.uniform(POOL_BOUNDS["x_min"], POOL_BOUNDS["x_max"]),
                    random.uniform(POOL_BOUNDS["y_min"], POOL_BOUNDS["y_max"]),
                    random.uniform(POOL_BOUNDS["z_min"], POOL_BOUNDS["z_max"])
                )
                cam_rot = unreal.Rotator(
                    roll=0.0,
                    pitch=random.uniform(-70.0, 0.0),
                    yaw=random.uniform(0.0, 360.0)
                )

                self.sample_data.append({
                    "frame_idx": i,
                    "target": None,
                    "cam_pos": cam_pos,
                    "cam_rot": cam_rot,
                    "is_negative": True
                })

                # Move ALL targets far underground for negative samples
                for t, track_data in target_tracks.items():
                    orig_loc = track_data['orig_loc']
                    orig_rot = track_data['orig_rot']
                    t_channels = track_data['channels']
                    t_channels[0].add_key(frame_time, orig_loc.x)
                    t_channels[1].add_key(frame_time, orig_loc.y)
                    t_channels[2].add_key(frame_time, -20000.0)
                    t_channels[3].add_key(frame_time, orig_rot.roll)
                    t_channels[4].add_key(frame_time, orig_rot.pitch)
                    t_channels[5].add_key(frame_time, orig_rot.yaw)

            else:
                target = target_assignments[positive_idx]
                positive_idx += 1

                # Generate randomized transforms for ALL targets this frame
                # Retry if any objects overlap (rejection sampling)
                for _collision_attempt in range(MAX_COLLISION_RETRIES):
                    frame_target_transforms = {}
                    for t in self.targets:
                        orig = target_tracks[t]
                        if RANDOMIZE_OBJECTS:
                            rand_loc = unreal.Vector(
                                orig['orig_loc'].x + random.uniform(-OBJECT_XY_RANGE_X, OBJECT_XY_RANGE_X),
                                orig['orig_loc'].y + random.uniform(-OBJECT_XY_RANGE_Y, OBJECT_XY_RANGE_Y),
                                orig['orig_loc'].z  # keep on surface
                            )
                            # Per-axis rotation lock: LockRoll/LockYaw/LockPitch tags
                            # freeze that axis to its original value
                            rand_rot = unreal.Rotator(
                                roll=orig['orig_rot'].roll if t.actor_has_tag("LockRoll") else random.uniform(-OBJECT_ROLL_RANGE, OBJECT_ROLL_RANGE),
                                pitch=orig['orig_rot'].pitch if t.actor_has_tag("LockPitch") else random.uniform(OBJECT_PITCH_MIN, OBJECT_PITCH_MAX),
                                yaw=orig['orig_rot'].yaw if t.actor_has_tag("LockYaw") else random.uniform(0, OBJECT_YAW_RANGE)
                            )
                        else:
                            rand_loc = orig['orig_loc']
                            rand_rot = orig['orig_rot']
                        frame_target_transforms[t] = {'loc': rand_loc, 'rot': rand_rot}

                    if not RANDOMIZE_OBJECTS or _check_no_overlap_2d(self.targets, frame_target_transforms, OBJECT_MIN_SEPARATION):
                        break
                else:
                    # All retries exhausted — fall back to original positions
                    unreal.log_warning(f"  Frame {i}: collision retry limit reached, using original positions")
                    frame_target_transforms = {
                        t: {'loc': target_tracks[t]['orig_loc'], 'rot': target_tracks[t]['orig_rot']}
                        for t in self.targets
                    }

                target_loc = frame_target_transforms[target]['loc']

                # Fetch DOPE_Bounds proxy box if it exists, else full actor bounds
                # Then shift bbox_center by the randomization offset for camera aiming
                use_custom_bounds = False
                orig_target_loc = target.get_actor_location()
                for comp in target.get_components_by_class(unreal.BoxComponent):
                    if comp.component_has_tag("DOPE_Bounds"):
                        bbox_center = comp.get_world_location()
                        use_custom_bounds = True
                        break

                if not use_custom_bounds:
                    bbox_center, _ = target.get_actor_bounds(False)

                if RANDOMIZE_OBJECTS:
                    bbox_center = unreal.Vector(
                        bbox_center.x + (target_loc.x - orig_target_loc.x),
                        bbox_center.y + (target_loc.y - orig_target_loc.y),
                        bbox_center.z + (target_loc.z - orig_target_loc.z)
                    )

                cam_pos = generate_clamped_position(target_loc)
                cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, bbox_center)

                if ENABLE_CAM_JITTER:
                    dist = math.sqrt((cam_pos.x - bbox_center.x)**2 +
                                     (cam_pos.y - bbox_center.y)**2 +
                                     (cam_pos.z - bbox_center.z)**2)
                    max_offset = dist * math.tan(math.radians(CAM_JITTER_MAX_PITCH))

                    margin = 0.10
                    jitter_scale = 1.0

                    for _attempt in range(4):
                        offset = max_offset * jitter_scale
                        look_at_point = unreal.Vector(
                            bbox_center.x + random.uniform(-offset, offset),
                            bbox_center.y + random.uniform(-offset, offset),
                            bbox_center.z + random.uniform(-offset * 0.5, offset * 0.5)
                        )
                        test_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, look_at_point)

                        test_transform = unreal.Transform(location=cam_pos, rotation=test_rot)
                        centroid_2d = project_point(bbox_center, test_transform, self.intrinsics)
                        if (centroid_2d != [-9999.0, -9999.0] and
                            RESOLUTION_X * margin < centroid_2d[0] < RESOLUTION_X * (1 - margin) and
                            RESOLUTION_Y * margin < centroid_2d[1] < RESOLUTION_Y * (1 - margin)):
                            cam_rot = test_rot
                            break
                        jitter_scale *= 0.5

                self.sample_data.append({
                    "frame_idx": i,
                    "target": target,
                    "cam_pos": cam_pos,
                    "cam_rot": cam_rot,
                    "is_negative": False
                })

                # Keyframe ALL targets with randomized transforms
                for t, track_data in target_tracks.items():
                    tf = frame_target_transforms[t]
                    t_channels = track_data['channels']
                    t_channels[0].add_key(frame_time, tf['loc'].x)
                    t_channels[1].add_key(frame_time, tf['loc'].y)
                    t_channels[2].add_key(frame_time, tf['loc'].z)
                    t_channels[3].add_key(frame_time, tf['rot'].roll)
                    t_channels[4].add_key(frame_time, tf['rot'].pitch)
                    t_channels[5].add_key(frame_time, tf['rot'].yaw)

            channels[0].add_key(frame_time, cam_pos.x)
            channels[1].add_key(frame_time, cam_pos.y)
            channels[2].add_key(frame_time, cam_pos.z)
            channels[3].add_key(frame_time, cam_rot.roll)
            channels[4].add_key(frame_time, cam_rot.pitch)
            channels[5].add_key(frame_time, cam_rot.yaw)

        for channel in channels:
            for key in channel.get_keys():
                key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)
                
        for t, t_data in target_tracks.items():
            for channel in t_data['channels']:
                for key in channel.get_keys():
                    key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)

        camera_cut_track = self._add_camera_cut_track(seq)
        if camera_cut_track:
            cam_binding_id = unreal.MovieSceneObjectBindingID()
            cam_binding_id.set_editor_property("guid", cam_binding.get_id())
            cut_section = camera_cut_track.add_section()
            cut_section.set_range(0, total_frames + 10)
            cut_section.set_camera_binding_id(cam_binding_id)

        return seq

    def _add_camera_cut_track(self, seq):
        movie_scene = seq.get_movie_scene()
        try:
            return movie_scene.add_camera_cut_track()
        except:
            try:
                return seq.add_track(unreal.MovieSceneCameraCutTrack)
            except:
                return None

    def _generate_all_labels(self):
        use_masks = USE_MASK_BBOX and HAS_CV2

        if use_masks:
            unreal.log("Generating YOLO labels (slow mask-based bounding boxes)...")
            self.capture_actor, self.render_target = setup_scene_capture(self.camera)
            if not self.capture_actor:
                unreal.log_warning("SceneCapture2D setup failed, falling back to AABB.")
                use_masks = False
        else:
            unreal.log("Generating YOLO labels (fast AABB projection)...")

        total_boxes = 0
        empty_frames = 0

        with unreal.ScopedSlowTask(self.total_samples, "Generating YOLO Labels...") as slow_task:
            slow_task.make_dialog(True) 

            for data in self.sample_data:
                if slow_task.should_cancel():
                    break

                slow_task.enter_progress_frame(1)

                i = data["frame_idx"]
                cam_pos = data["cam_pos"]
                cam_rot = data["cam_rot"]
                is_negative = data.get("is_negative", False)
                
                label_path = os.path.join(self.staging_labels, f"{i:06d}.txt")

                if is_negative:
                    with open(label_path, 'w') as f:
                        f.write("")
                    empty_frames += 1
                    continue

                label_lines = []
                cam_transform = unreal.Transform(location=cam_pos, rotation=cam_rot)
                for target in self.targets:
                    if use_masks and self.capture_actor:
                        bbox = capture_actor_mask_bbox(
                            self.capture_actor, self.render_target, self.camera,
                            cam_pos, cam_rot, target, i
                        )
                    else:
                        bbox = _get_2d_bbox_fallback(target, cam_transform, self.intrinsics)

                    if bbox:
                        class_id = self.class_map[target.get_actor_label()]
                        x_center, y_center, width, height = bbox
                        label_lines.append(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )
                        total_boxes += 1

                with open(label_path, 'w') as f:
                    f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

                if not label_lines:
                    empty_frames += 1

                if (i + 1) % 10 == 0:
                    unreal.log(f"  Progress: {i + 1}/{self.total_samples} frames")

        if self.capture_actor:
            self.capture_actor.destroy_actor()
            self.capture_actor = None
            
        if unreal.EditorAssetLibrary.does_asset_exist(RT_ASSET_PATH):
            unreal.EditorAssetLibrary.delete_asset(RT_ASSET_PATH)

        unreal.log(f"  Labels complete: {total_boxes} boxes across "
                   f"{self.total_samples} frames ({empty_frames} empty frames)")

    def _render(self, sequence):
        global global_executor
        unreal.log("Starting MRQ render...")

        mrq = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        queue = mrq.get_queue()
        queue.delete_all_jobs()

        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
        job.map = unreal.SoftObjectPath(get_world().get_path_name())
        job.job_name = "YOLO_Dataset"

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
        output.custom_end_frame = self.total_samples * 2
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
            "r.TemporalAACurrentFrameWeight 1.0",
            "r.MotionBlurQuality 0",
            "r.MotionBlur.Max 0",
            "r.DepthOfFieldQuality 0",
            "r.DefaultFeature.AntiAliasing 1",
            "r.VolumetricFog.TemporalReprojection 0",
            "r.ScreenPercentage 100",
        ]

        config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)

        global_executor = unreal.MoviePipelinePIEExecutor()

        class_map = self.class_map 

        def on_finished(executor, success):
            global global_executor
            unreal.log("=" * 60)
            unreal.log(f"RENDER COMPLETE! Success: {success}")
            unreal.log("Cleaning up gap frames...")
            cleanup_and_renumber_frames()
            unreal.log("Splitting into train/val sets...")
            split_dataset(OUTPUT_FOLDER, VAL_SPLIT_RATIO)
            unreal.log("Generating data.yaml...")
            generate_data_yaml(OUTPUT_FOLDER, class_map)
            unreal.log(f"Output: {OUTPUT_FOLDER}")
            unreal.log("Dataset is ready for: yolo detect train data=data.yaml")
            unreal.log("=" * 60)
            global_executor = None

        global_executor.on_executor_finished_delegate.add_callable(on_finished)
        mrq.render_queue_with_executor_instance(global_executor)


def cleanup_and_renumber_frames():
    images_folder = os.path.join(OUTPUT_FOLDER, "_staging", "images")
    png_files = sorted(glob.glob(os.path.join(images_folder, "**", "*.png"), recursive=True))

    deleted_count = 0
    renamed_count = 0

    for png_path in png_files:
        filename = os.path.basename(png_path)
        name_part = os.path.splitext(filename)[0]

        try:
            frame_num = int(name_part)
        except ValueError:
            continue

        if frame_num % 2 == 1:
            os.remove(png_path)
            deleted_count += 1
        else:
            new_num = frame_num // 2
            new_path = os.path.join(images_folder, f"{new_num:06d}.png")
            if png_path != new_path:
                os.rename(png_path, new_path)
                renamed_count += 1

    for item in os.listdir(images_folder):
        item_path = os.path.join(images_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

    unreal.log(f"  Deleted {deleted_count} gap frames, renamed {renamed_count} frames")


def split_dataset(output_folder, val_ratio=0.2):
    staging_images = os.path.join(output_folder, "_staging", "images")
    staging_labels = os.path.join(output_folder, "_staging", "labels")

    all_images = sorted(glob.glob(os.path.join(staging_images, "*.png")))

    if not all_images:
        unreal.log_warning("No images found in staging directory for split!")
        return

    random.shuffle(all_images)
    split_idx = max(1, int(len(all_images) * (1 - val_ratio)))
    splits = {
        "train": all_images[:split_idx],
        "val":   all_images[split_idx:],
    }

    for split_name, img_list in splits.items():
        split_img_dir = os.path.join(output_folder, split_name, "images")
        split_lbl_dir = os.path.join(output_folder, split_name, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        for img_path in img_list:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            shutil.move(img_path, os.path.join(split_img_dir, f"{basename}.png"))
            lbl_path = os.path.join(staging_labels, f"{basename}.txt")
            if os.path.exists(lbl_path):
                shutil.move(lbl_path, os.path.join(split_lbl_dir, f"{basename}.txt"))

        unreal.log(f"  {split_name}: {len(img_list)} samples")

    staging_dir = os.path.join(output_folder, "_staging")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)


def generate_data_yaml(output_folder, class_map):
    sorted_classes = sorted(class_map.items(), key=lambda x: x[1])
    names = {idx: name for name, idx in sorted_classes}

    yaml_path = os.path.join(output_folder, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_folder.rstrip('/').rstrip(chr(92))}\\n")
        f.write("train: train/images\\n")
        f.write("val: val/images\\n")
        f.write(f"nc: {len(names)}\\n")
        f.write("names:\\n")
        for idx in sorted(names.keys()):
            f.write(f"  {idx}: {names[idx]}\\n")

    unreal.log(f"  data.yaml saved to {yaml_path}")
    unreal.log(f"  Classes ({len(names)}): {names}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if 'yolo_gen' in dir():
    del yolo_gen

if 'global_executor' in dir() and global_executor:
    global_executor = None

yolo_gen = YOLODatasetGenerator()
