"""
UE5.7.1 DOPE Dataset Generator - V4 (Visibility-Aware + Camera Jitter)
Generates synthetic training data for Deep Object Pose Estimation (DOPE)

V4 improvements over V3:
- Multi-object annotation: ALL targets annotated per frame (not just aimed target)
- Visibility measurement: SceneCapture two-pass differential to measure true visibility
- DOPE Rules compliance:
    Rule 1: Full 3D cuboid always projected (never shrunk)
    Rule 2: Objects with <20% visibility (occluded) are skipped
    Rule 3: Objects whose centroid falls outside image bounds are skipped
    Rule 4: Objects with fewer than MIN_VISIBLE_PIXELS are skipped (gradient noise)
- Camera jitter: Random pitch/yaw offset to avoid center bias
- Camera configuration lock: Sensor matching, focus breathing disabled
- Stale SceneCapture2D actor cleanup

Prerequisites:
    Install opencv in UE5's Python:
        <UE5_ROOT>/Engine/Binaries/ThirdParty/Python3/Win64/python.exe -m pip install opencv-python-headless
"""

import unreal
import json
import math
import os
import shutil
import random
import glob

# Try to import cv2/numpy for visibility measurement
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# Scene Tags
TARGET_TAG = "TrainObject"
CAMERA_TAG = "AUV_Camera"

# Output Settings
OUTPUT_FOLDER = "D:/UE5_DOPE_Data/"
SEQUENCE_PATH = "/Game/Generated/DOPESequence"
SAMPLES_PER_OBJECT = 10  # Each target gets this many aimed frames

# Camera Movement
MIN_DISTANCE = 100.0   # cm
MAX_DISTANCE = 400.0   # cm

# Resolution
RESOLUTION_X = 1920
RESOLUTION_Y = 1080

# Camera Intrinsics (from your camera settings)
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 20.25  # Matches 16:9 aspect ratio (36 / 1.777...)
FOCAL_LENGTH_MM = 30.0

# Pool Bounds
POOL_BOUNDS = {
    "x_min": -1776.0, "x_max": 989.0,
    "y_min": -3992.0, "y_max": 690.0,
    "z_min": -1841.0, "z_max": -1360.0
}

# Render Settings - Aggressive anti-ghosting
WARMUP_FRAMES = 64
SPATIAL_SAMPLES = 4
TEMPORAL_SAMPLES = 1  # CRITICAL: Keep at 1 to avoid ghosting

# Mask capture resolution (for visibility measurement)
# Lower = faster pixel reads, still accurate enough for pixel counting
MASK_RESOLUTION_X = 480
MASK_RESOLUTION_Y = 270

# Render target asset path (created as a UE asset in content browser)
RT_ASSET_PATH = "/Game/Generated/DOPEMaskRT"

# Visibility thresholds (DOPE best practices)
MIN_VISIBILITY_RATIO = 0.20    # Rule 2: Skip if <20% visible (occlusion)
MIN_VISIBLE_PIXELS = 15        # Rule 4: Skip if fewer visible pixels (at mask resolution)

# Camera jitter (random tilt to avoid center bias)
ENABLE_CAM_JITTER = False       # Set to False to always center the target
CAM_JITTER_MAX_PITCH = 15.0    # degrees, ±range for pitch offset
CAM_JITTER_MAX_YAW = 15.0      # degrees, ±range for yaw offset

# Negative samples (frames with no objects, helps model learn background)
NEGATIVE_SAMPLE_RATIO = 0.10   # 10% of samples will be negative

# Save debug mask images for frame 0 (for diagnosing visibility measurement)
SAVE_DEBUG_MASKS = False

# Global reference to prevent garbage collection
global_executor = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_world():
    """Get the editor world using the API."""
    try:
        subsys = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        return subsys.get_editor_world()
    except (AttributeError, Exception):
        return unreal.EditorLevelLibrary.get_editor_world()


def calculate_intrinsics():
    """Calculate camera intrinsic matrix from physical parameters."""
    px_per_mm_x = RESOLUTION_X / SENSOR_WIDTH_MM
    px_per_mm_y = RESOLUTION_Y / SENSOR_HEIGHT_MM

    fx = FOCAL_LENGTH_MM * px_per_mm_x
    fy = FOCAL_LENGTH_MM * px_per_mm_y
    cx = RESOLUTION_X / 2.0
    cy = RESOLUTION_Y / 2.0

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def ue_to_opencv_location(ue_location, cam_transform):
    """Transform location from UE5 world space to camera space (OpenCV convention)."""
    cam_inv = cam_transform.inverse()
    local_pos = unreal.MathLibrary.transform_location(cam_inv, ue_location)

    # UE5 (X-fwd, Y-right, Z-up) -> OpenCV (X-right, Y-down, Z-fwd)
    return [
        local_pos.y / 100.0,   # UE Y -> CV X (cm->m)
        -local_pos.z / 100.0,  # UE Z -> CV -Y (cm->m)
        local_pos.x / 100.0    # UE X -> CV Z (cm->m)
    ]


def ue_rotation_to_quaternion_xyzw(obj_rot, cam_transform):
    """Convert UE5 rotation to quaternion in camera frame (XYZW format)."""
    cam_inv = cam_transform.inverse()
    cam_rot_inv = cam_inv.rotation

    obj_quat = obj_rot.quaternion()
    relative_quat = cam_rot_inv * obj_quat

    length = math.sqrt(relative_quat.x**2 + relative_quat.y**2 +
                       relative_quat.z**2 + relative_quat.w**2)
    if length > 0:
        qx = relative_quat.x / length
        qy = relative_quat.y / length
        qz = relative_quat.z / length
        qw = relative_quat.w / length
    else:
        qx, qy, qz, qw = 0, 0, 0, 1

    # UE5 -> OpenCV coordinate system transform
    return [qy, -qz, qx, qw]


def get_cuboid_corners(actor):
    """Get 9 cuboid points (8 corners + centroid) in DOPE order."""
    origin, extent = actor.get_actor_bounds(False)
    ex, ey, ez = extent.x, extent.y, extent.z
    ox, oy, oz = origin.x, origin.y, origin.z

    raw = [
        unreal.Vector(ox + ex, oy + ey, oz + ez),  # 0
        unreal.Vector(ox + ex, oy + ey, oz - ez),  # 1
        unreal.Vector(ox + ex, oy - ey, oz + ez),  # 2
        unreal.Vector(ox + ex, oy - ey, oz - ez),  # 3
        unreal.Vector(ox - ex, oy + ey, oz + ez),  # 4
        unreal.Vector(ox - ex, oy + ey, oz - ez),  # 5
        unreal.Vector(ox - ex, oy - ey, oz + ez),  # 6
        unreal.Vector(ox - ex, oy - ey, oz - ez),  # 7
    ]

    # DOPE ordering (matches BlenderProc)
    dope_order = [5, 1, 2, 6, 4, 0, 3, 7]
    corners = [raw[i] for i in dope_order]
    corners.append(origin)  # Centroid
    return corners


def project_point(world_pt, cam_transform, intrinsics):
    """Project 3D point to 2D image coordinates."""
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
    """Generate random camera position within bounds."""
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
# SCENE CAPTURE VISIBILITY MEASUREMENT
# =============================================================================

def create_render_target_asset(width, height):
    """Create a render target asset for visibility captures."""
    # Clean up any existing asset
    if unreal.EditorAssetLibrary.does_asset_exist(RT_ASSET_PATH):
        unreal.EditorAssetLibrary.delete_asset(RT_ASSET_PATH)

    world = get_world()

    # Strategy 1: RenderingLibrary.create_render_target2d()
    try:
        rt = unreal.RenderingLibrary.create_render_target2d(
            world,
            width=width,
            height=height,
            format=unreal.TextureRenderTargetFormat.RTF_RGBA8,
            clear_color=unreal.LinearColor(0, 0, 0, 1)
        )
        if rt:
            unreal.log(f"  Render target created via RenderingLibrary: {width}x{height} RGBA8")
            return rt
    except (AttributeError, Exception) as e:
        unreal.log_warning(f"  RenderingLibrary.create_render_target2d failed: {e}")

    # Strategy 2: Fallback — create via CanvasRenderTarget2DFactoryNew
    try:
        pkg_path, asset_name = RT_ASSET_PATH.rsplit('/', 1)
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
        factory = unreal.CanvasRenderTarget2DFactoryNew()
        rt = asset_tools.create_asset(
            asset_name, pkg_path,
            unreal.CanvasRenderTarget2D, factory
        )
        if rt:
            rt.set_editor_property('size_x', width)
            rt.set_editor_property('size_y', height)
            rt.set_editor_property('render_target_format',
                                   unreal.TextureRenderTargetFormat.RTF_RGBA8)
            rt.set_editor_property('clear_color', unreal.LinearColor(0, 0, 0, 1))
            unreal.log(f"  Render target created via factory fallback: {width}x{height} RGBA8")
            return rt
    except (AttributeError, Exception) as e:
        unreal.log_warning(f"  Factory fallback failed: {e}")

    unreal.log_error("Failed to create render target!")
    return None


def setup_scene_capture(camera_actor):
    """
    Spawn a SceneCapture2D actor and render target for visibility measurement.

    Uses PRM_RENDER_SCENE_NORMALLY so the full scene (including occluding
    geometry) is rendered. Two-pass differential is used per target.

    Args:
        camera_actor: The CineCameraActor to match FOV from

    Returns:
        (capture_actor, render_target) or (None, None) on failure
    """
    rt = create_render_target_asset(MASK_RESOLUTION_X, MASK_RESOLUTION_Y)
    if not rt:
        return None, None

    # Read the actual FOV from the CineCameraComponent
    try:
        cine_comp = camera_actor.get_cine_camera_component()
        fov_degrees = cine_comp.field_of_view
    except (AttributeError, Exception):
        fov_degrees = 2.0 * math.atan(SENSOR_WIDTH_MM / (2.0 * FOCAL_LENGTH_MM))
        fov_degrees = fov_degrees * (180.0 / math.pi)

    # Spawn SceneCapture2D actor
    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    capture_actor = subsys.spawn_actor_from_class(
        unreal.SceneCapture2D,
        unreal.Vector(0, 0, 0),
        unreal.Rotator(0, 0, 0)
    )
    if not capture_actor:
        unreal.log_error("Failed to spawn SceneCapture2D actor!")
        return None, None

    cc = capture_actor.capture_component2d
    cc.texture_target = rt
    cc.set_editor_property('primitive_render_mode',
                           unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES)
    cc.set_editor_property('capture_source',
                           unreal.SceneCaptureSource.SCS_BASE_COLOR)
    cc.set_editor_property('fov_angle', fov_degrees)
    cc.set_editor_property('capture_every_frame', False)
    cc.set_editor_property('capture_on_movement', False)

    unreal.log(f"  SceneCapture2D ready (FOV: {fov_degrees:.1f} deg, "
               f"mask {MASK_RESOLUTION_X}x{MASK_RESOLUTION_Y})")

    return capture_actor, rt


def _read_render_target_as_numpy(render_target):
    """
    Read the render target contents into a numpy array (H, W, 3) uint8 BGR.
    """
    world = get_world()
    colors = unreal.RenderingLibrary.read_render_target(
        world, render_target, normalize=True
    )
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


def _position_capture(capture_actor, cine_camera, cam_pos, cam_rot):
    """
    Position the CineCamera and SceneCapture at the given camera pose.
    Uses the CineCameraComponent's actual world transform for exact alignment.
    """
    cc = capture_actor.capture_component2d

    # Move the CineCamera to this position
    cine_camera.set_actor_location_and_rotation(
        cam_pos, cam_rot, False, True  # sweep=False, teleport=True
    )

    # Read the CineCameraComponent's actual world transform
    try:
        cine_comp = cine_camera.get_cine_camera_component()
        actual_pos = cine_comp.get_world_location()
        actual_rot = cine_comp.get_world_rotation()
    except (AttributeError, Exception):
        actual_pos = cam_pos
        actual_rot = cam_rot

    # Position SceneCapture at the exact camera transform
    cc.set_world_location_and_rotation(
        actual_pos, actual_rot, False, True
    )


def _diff_pixel_count(img_a, img_b, threshold=5):
    """Compute the number of pixels that differ between two images."""
    diff = cv2.absdiff(img_a, img_b)
    diff_gray = np.max(diff, axis=2)
    return int(np.count_nonzero(diff_gray > threshold))


def measure_all_visibilities(capture_actor, render_target, cine_camera,
                             cam_pos, cam_rot, targets, frame_idx):
    """
    Measure visibility for all targets at the given camera position.

    Multi-pass SceneCapture approach:
      Phase 1 (solo measurements - total unoccluded pixel count per target):
        - Hide all targets → capture empty_bg
        - For each target: show only that target → capture → total_pixels
      Phase 2 (visible measurements - actually visible pixel count per target):
        - Show all targets → capture full_scene
        - For each target: hide that target → capture → visible_pixels

    Returns:
        dict mapping target_actor → {
            'visible_pixels': int,
            'total_pixels': int,
            'visibility': float  # visible / total
        }
    """
    cc = capture_actor.capture_component2d
    _position_capture(capture_actor, cine_camera, cam_pos, cam_rot)

    # ── Phase 1: Solo measurements ──────────────────────────────
    # Hide all targets to get empty scene background
    for t in targets:
        t.set_actor_hidden_in_game(True)
    cc.capture_scene()
    empty_bg = _read_render_target_as_numpy(render_target)

    # Capture each target solo (no inter-target occlusion)
    solo_pixels = {}
    for target in targets:
        target.set_actor_hidden_in_game(False)
        cc.capture_scene()
        solo_fg = _read_render_target_as_numpy(render_target)
        target.set_actor_hidden_in_game(True)

        if empty_bg is not None and solo_fg is not None:
            solo_pixels[target] = _diff_pixel_count(solo_fg, empty_bg)
        else:
            solo_pixels[target] = 0

    # ── Phase 2: Visible measurements ───────────────────────────
    # Show all targets for full scene
    for t in targets:
        t.set_actor_hidden_in_game(False)
    cc.capture_scene()
    full_scene = _read_render_target_as_numpy(render_target)

    # Hide each target one at a time to see what disappears
    visible_pixels = {}
    for target in targets:
        target.set_actor_hidden_in_game(True)
        cc.capture_scene()
        no_target = _read_render_target_as_numpy(render_target)
        target.set_actor_hidden_in_game(False)

        if full_scene is not None and no_target is not None:
            visible_pixels[target] = _diff_pixel_count(full_scene, no_target)
        else:
            visible_pixels[target] = 0

    # ── Save debug images for frame 0 ──────────────────────────
    if SAVE_DEBUG_MASKS and frame_idx == 0 and empty_bg is not None and full_scene is not None:
        debug_dir = os.path.join(OUTPUT_FOLDER, "_debug_masks")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "empty_bg.png"), empty_bg)
        cv2.imwrite(os.path.join(debug_dir, "full_scene.png"), full_scene)
        unreal.log(f"  Debug masks saved for frame 0 → {debug_dir}")

    # ── Compute visibility ratios ───────────────────────────────
    results = {}
    for target in targets:
        total = solo_pixels.get(target, 0)
        visible = visible_pixels.get(target, 0)
        ratio = float(visible) / float(total) if total > 0 else 0.0
        results[target] = {
            'visible_pixels': visible,
            'total_pixels': total,
            'visibility': ratio
        }

    return results


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class DOPEDatasetGenerator:
    def __init__(self):
        self.targets = []
        self.camera = None
        self.intrinsics = calculate_intrinsics()
        self.sample_data = []
        self.capture_actor = None
        self.render_target = None

        unreal.log("=" * 60)
        unreal.log("UE5.7 DOPE DATASET GENERATOR V4")
        unreal.log("  (Visibility-Aware + Camera Jitter)")
        unreal.log("=" * 60)

        if not HAS_CV2:
            unreal.log_warning(
                "WARNING: cv2 not found — visibility measurement disabled.\n"
                "  Install via: <UE5>/Engine/Binaries/ThirdParty/Python3/Win64/python.exe "
                "-m pip install opencv-python-headless\n"
                "  Rules 2 & 4 (occlusion/minimum feature) will NOT be enforced."
            )

        # Setup output folder
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
        os.makedirs(OUTPUT_FOLDER)

        # Find actors
        self._find_actors()
        if not self.camera or not self.targets:
            unreal.log_error("ERROR: Camera or targets not found!")
            return

        unreal.log(f"Found {len(self.targets)} targets, Camera: {self.camera.get_actor_label()}")

        # Compute total samples: SAMPLES_PER_OBJECT * num_targets + negatives
        num_positive = SAMPLES_PER_OBJECT * len(self.targets)
        num_negative = int(num_positive * NEGATIVE_SAMPLE_RATIO / (1 - NEGATIVE_SAMPLE_RATIO))
        self.total_samples = num_positive + num_negative

        unreal.log(f"  {SAMPLES_PER_OBJECT} samples/object × {len(self.targets)} objects "
                   f"= {num_positive} positive + {num_negative} negative "
                   f"= {self.total_samples} total")

        self._configure_camera()

        # Log DOPE rules configuration
        if HAS_CV2:
            unreal.log(f"  Visibility Rules: min_ratio={MIN_VISIBILITY_RATIO:.0%}, "
                       f"min_pixels={MIN_VISIBLE_PIXELS}")
        unreal.log(f"  Camera Jitter: ±{CAM_JITTER_MAX_PITCH}° pitch, "
                   f"±{CAM_JITTER_MAX_YAW}° yaw")

        # Run pipeline
        self._run()

    def _configure_camera(self):
        """Force camera settings to match render output exactly to prevent drift."""
        if not self.camera:
            return

        cine_comp = self.camera.get_cine_camera_component()

        cine_comp.filmback.sensor_width = SENSOR_WIDTH_MM
        cine_comp.filmback.sensor_height = SENSOR_HEIGHT_MM
        cine_comp.current_focal_length = FOCAL_LENGTH_MM
        cine_comp.focus_settings.focus_method = unreal.CameraFocusMethod.DISABLE

        unreal.log(f"  Camera Configured: Sensor {SENSOR_WIDTH_MM}x{SENSOR_HEIGHT_MM}mm "
                   f"(16:9), FL {FOCAL_LENGTH_MM}mm, Focus Breathing DISABLED")

    def _find_actors(self):
        """Find tagged actors and clean up stale SceneCapture2D actors."""
        subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        stale_count = 0
        for actor in subsys.get_all_level_actors():
            if actor.actor_has_tag(TARGET_TAG):
                self.targets.append(actor)
            elif actor.actor_has_tag(CAMERA_TAG):
                self.camera = actor
            elif actor.get_class().get_name() == "SceneCapture2D":
                actor.destroy_actor()
                stale_count += 1
        if stale_count:
            unreal.log(f"  Cleaned up {stale_count} stale SceneCapture2D actor(s)")

    def _run(self):
        """Main pipeline."""
        sequence = self._create_sequence()
        if not sequence:
            return

        self._generate_all_jsons()

        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
        self._render(sequence)

    def _create_sequence(self):
        """Create level sequence with camera keyframes (including jitter)."""
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
        total_samples = self.total_samples
        total_frames = total_samples * frames_per_sample

        transform_section.set_range(0, total_frames + 10)
        seq.set_playback_start(0)
        seq.set_playback_end(total_frames)

        channels = transform_section.get_all_channels()

        num_positive = SAMPLES_PER_OBJECT * len(self.targets)
        num_negative = total_samples - num_positive

        # Build balanced target list: each target gets exactly SAMPLES_PER_OBJECT frames
        # then shuffle and interleave with negatives
        target_assignments = []
        for target in self.targets:
            target_assignments.extend([target] * SAMPLES_PER_OBJECT)
        random.shuffle(target_assignments)

        # Build frame type list: positives then negatives, then shuffle
        frame_types = ['positive'] * num_positive + ['negative'] * num_negative
        random.shuffle(frame_types)

        unreal.log(f"Generating camera positions (with jitter)...")
        unreal.log(f"  {num_positive} positive ({SAMPLES_PER_OBJECT}/object) + {num_negative} negative")

        positive_idx = 0  # Index into target_assignments
        for i in range(total_samples):
            is_negative = (frame_types[i] == 'negative')

            if is_negative:
                # Negative sample: random position in pool, fixed rotation
                # Camera looks away from pool surface (not downward)
                cam_pos = unreal.Vector(
                    random.uniform(POOL_BOUNDS["x_min"], POOL_BOUNDS["x_max"]),
                    random.uniform(POOL_BOUNDS["y_min"], POOL_BOUNDS["y_max"]),
                    random.uniform(POOL_BOUNDS["z_min"], POOL_BOUNDS["z_max"])
                )
                # Roll=0, Pitch=0 to -70 (looking upward/horizontal), Yaw=random
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
                    "target_loc": None,
                    "target_rot": None,
                    "is_negative": True
                })
            else:
                # Positive sample: balanced round-robin target assignment
                target = target_assignments[positive_idx]
                positive_idx += 1
                target_loc = target.get_actor_location()
                target_rot = target.get_actor_rotation()
                bbox_center, bbox_extents = target.get_actor_bounds(False)

                cam_pos = generate_clamped_position(target_loc)

                cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, bbox_center)

                # Apply camera jitter if enabled
                if ENABLE_CAM_JITTER:
                    # Camera jitter via look-at-point offset (avoids gimbal lock):
                    # Instead of modifying the Rotator (which causes gimbal issues),
                    # we offset the look-at target point and let find_look_at_rotation
                    # naturally compute a correct, upright camera rotation.
                    # The offset is proportional to distance for consistent angular displacement.
                    dist = math.sqrt((cam_pos.x - bbox_center.x)**2 +
                                     (cam_pos.y - bbox_center.y)**2 +
                                     (cam_pos.z - bbox_center.z)**2)
                    max_offset = dist * math.tan(math.radians(CAM_JITTER_MAX_PITCH))

                    margin = 0.10  # 10% margin from edges
                    jitter_scale = 1.0

                    for _attempt in range(4):
                        offset = max_offset * jitter_scale
                        look_at_point = unreal.Vector(
                            bbox_center.x + random.uniform(-offset, offset),
                            bbox_center.y + random.uniform(-offset, offset),
                            bbox_center.z + random.uniform(-offset * 0.5, offset * 0.5)  # less vertical
                        )
                        test_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, look_at_point)

                        # Validate: aimed target stays in-frame
                        test_transform = unreal.Transform(location=cam_pos, rotation=test_rot)
                        centroid_2d = project_point(bbox_center, test_transform, self.intrinsics)
                        if (centroid_2d != [-9999.0, -9999.0] and
                            RESOLUTION_X * margin < centroid_2d[0] < RESOLUTION_X * (1 - margin) and
                            RESOLUTION_Y * margin < centroid_2d[1] < RESOLUTION_Y * (1 - margin)):
                            cam_rot = test_rot
                            break
                        # Too much jitter — halve and retry
                        jitter_scale *= 0.5

                self.sample_data.append({
                    "frame_idx": i,
                    "target": target,
                    "cam_pos": cam_pos,
                    "cam_rot": cam_rot,
                    "target_loc": target_loc,
                    "target_rot": target_rot,
                    "is_negative": False
                })

            frame_num = i * frames_per_sample
            frame_time = unreal.FrameNumber(frame_num)

            channels[0].add_key(frame_time, cam_pos.x)
            channels[1].add_key(frame_time, cam_pos.y)
            channels[2].add_key(frame_time, cam_pos.z)
            channels[3].add_key(frame_time, cam_rot.roll)
            channels[4].add_key(frame_time, cam_rot.pitch)
            channels[5].add_key(frame_time, cam_rot.yaw)

        # Set CONSTANT interpolation (instant teleport, no blending!)
        for channel in channels:
            for key in channel.get_keys():
                key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)

        # Camera cut track
        camera_cut_track = self._add_camera_cut_track(seq)
        if camera_cut_track:
            cam_binding_id = unreal.MovieSceneObjectBindingID()
            cam_binding_id.set_editor_property("guid", cam_binding.get_id())
            cut_section = camera_cut_track.add_section()
            cut_section.set_range(0, total_frames + 10)
            cut_section.set_camera_binding_id(cam_binding_id)

        return seq

    def _add_camera_cut_track(self, seq):
        """Safely add camera cut track."""
        movie_scene = seq.get_movie_scene()
        try:
            return movie_scene.add_camera_cut_track()
        except:
            try:
                return seq.add_track(unreal.MovieSceneCameraCutTrack)
            except:
                try:
                    return movie_scene.add_track(unreal.MovieSceneCameraCutTrack)
                except:
                    return None

    def _generate_all_jsons(self):
        """Generate all JSON files with visibility-aware filtering.

        For each frame, ALL target actors are tested. Objects are filtered by:
          Rule 2: visibility > MIN_VISIBILITY_RATIO (occlusion check)
          Rule 3: centroid must be within image bounds (truncation check)
          Rule 4: visible_pixels > MIN_VISIBLE_PIXELS (minimum feature check)

        The full 3D cuboid is ALWAYS projected without modification (Rule 1).
        """
        unreal.log("Generating JSON files...")

        # Setup SceneCapture for visibility measurement (if cv2 available)
        use_visibility = HAS_CV2
        if use_visibility:
            self.capture_actor, self.render_target = setup_scene_capture(self.camera)
            if not self.capture_actor:
                unreal.log_warning("SceneCapture setup failed! Proceeding without visibility checks.")
                use_visibility = False

        total_objects = 0
        total_negative = 0
        skipped_behind = 0
        skipped_truncation = 0
        skipped_occlusion = 0
        skipped_min_feature = 0

        with unreal.ScopedSlowTask(self.total_samples, "Generating DOPE JSON Files...") as slow_task:
            slow_task.make_dialog(True)

            for data in self.sample_data:
                if slow_task.should_cancel():
                    break

                slow_task.enter_progress_frame(1)

                i = data["frame_idx"]
                cam_pos = data["cam_pos"]
                cam_rot = data["cam_rot"]
                is_negative = data.get("is_negative", False)

                # ── Negative sample: empty annotation, hide all targets ──
                if is_negative:
                    # Hide all targets so they don't appear in the rendered frame
                    for t in self.targets:
                        t.set_actor_hidden_in_game(True)

                    json_data = {
                        "camera_data": {
                            "width": RESOLUTION_X,
                            "height": RESOLUTION_Y,
                            "intrinsics": self.intrinsics
                        },
                        "objects": []
                    }
                    json_path = os.path.join(OUTPUT_FOLDER, f"{i:06d}.json")
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f, indent=4)
                    total_negative += 1

                    # Restore targets visibility for subsequent positive frames
                    for t in self.targets:
                        t.set_actor_hidden_in_game(False)

                    if (i + 1) % 10 == 0:
                        unreal.log(f"  Progress: {i + 1}/{self.total_samples} JSON files")
                    continue

                # ── Positive sample: normal DOPE annotation ──
                cam_transform = unreal.Transform(location=cam_pos, rotation=cam_rot)

                # Measure visibility for all targets at this camera position
                visibility_data = {}
                if use_visibility:
                    visibility_data = measure_all_visibilities(
                        self.capture_actor, self.render_target, self.camera,
                        cam_pos, cam_rot, self.targets, i
                    )

                # Annotate all targets, applying DOPE rules
                objects_list = []
                for target in self.targets:
                    target_loc = target.get_actor_location()
                    target_rot = target.get_actor_rotation()

                    # Rule 1: Always project the FULL 3D cuboid (never shrink)
                    corners = get_cuboid_corners(target)
                    projected = [project_point(c, cam_transform, self.intrinsics) for c in corners]

                    # Centroid is the last projected point (index 8)
                    centroid_2d = projected[-1]

                    # Check: centroid behind camera
                    actor_name = target.get_actor_label()
                    if centroid_2d == [-9999.0, -9999.0]:
                        skipped_behind += 1
                        if i < 5:  # Log details for first 5 frames
                            unreal.log(f"    Frame {i}: SKIP '{actor_name}' — centroid behind camera")
                        continue

                    # Rule 3: Centroid must be within image bounds (truncation check)
                    cx_px, cy_px = centroid_2d
                    if cx_px < 0 or cx_px > RESOLUTION_X or cy_px < 0 or cy_px > RESOLUTION_Y:
                        skipped_truncation += 1
                        if i < 5:
                            unreal.log(f"    Frame {i}: SKIP '{actor_name}' — centroid out of frame "
                                       f"({cx_px:.0f}, {cy_px:.0f})")
                        continue

                    # Rules 2 & 4: Visibility checks (requires cv2)
                    visibility = 1.0  # Default if no measurement
                    if use_visibility and target in visibility_data:
                        vis = visibility_data[target]
                        visibility = vis['visibility']

                        # Rule 4: Minimum feature threshold
                        if vis['visible_pixels'] < MIN_VISIBLE_PIXELS:
                            skipped_min_feature += 1
                            unreal.log(f"    Frame {i}: SKIP '{actor_name}' — Rule 4: "
                                       f"only {vis['visible_pixels']}px visible "
                                       f"(min: {MIN_VISIBLE_PIXELS})")
                            continue

                        # Rule 2: Occlusion threshold
                        if visibility < MIN_VISIBILITY_RATIO:
                            skipped_occlusion += 1
                            unreal.log(f"    Frame {i}: SKIP '{actor_name}' — Rule 2: "
                                       f"visibility {visibility:.1%} "
                                       f"({vis['visible_pixels']}/{vis['total_pixels']}px, "
                                       f"min: {MIN_VISIBILITY_RATIO:.0%})")
                            continue

                    objects_list.append({
                        "class": target.get_actor_label(),
                        "name": f"{target.get_actor_label()}_{i:03d}",
                        "visibility": round(visibility, 3),
                        "location": ue_to_opencv_location(target_loc, cam_transform),
                        "quaternion_xyzw": ue_rotation_to_quaternion_xyzw(target_rot, cam_transform),
                        "projected_cuboid": projected
                    })
                    total_objects += 1

                # Write JSON
                json_data = {
                    "camera_data": {
                        "width": RESOLUTION_X,
                        "height": RESOLUTION_Y,
                        "intrinsics": self.intrinsics
                    },
                    "objects": objects_list
                }

                json_path = os.path.join(OUTPUT_FOLDER, f"{i:06d}.json")
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=4)

                if (i + 1) % 10 == 0:
                    unreal.log(f"  Progress: {i + 1}/{self.total_samples} JSON files")

        # Cleanup SceneCapture
        if self.capture_actor:
            self.capture_actor.destroy_actor()
            self.capture_actor = None
        if unreal.EditorAssetLibrary.does_asset_exist(RT_ASSET_PATH):
            unreal.EditorAssetLibrary.delete_asset(RT_ASSET_PATH)

        unreal.log(f"  Generated {self.total_samples} JSON files: "
                   f"{self.total_samples - total_negative} positive ({total_objects} object annotations), "
                   f"{total_negative} negative")
        unreal.log(f"  Filtered: {skipped_behind} behind camera, "
                   f"{skipped_truncation} centroid out of frame (Rule 3), "
                   f"{skipped_occlusion} occluded <{MIN_VISIBILITY_RATIO:.0%} (Rule 2), "
                   f"{skipped_min_feature} below {MIN_VISIBLE_PIXELS}px (Rule 4)")

    def _render(self, sequence):
        """Execute MRQ render with aggressive anti-ghosting settings."""
        global global_executor

        unreal.log("Starting MRQ render...")

        mrq = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        queue = mrq.get_queue()
        queue.delete_all_jobs()

        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
        job.map = unreal.SoftObjectPath(get_world().get_path_name())
        job.job_name = "DOPE_Dataset"

        config = job.get_configuration()

        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)

        output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        output.output_directory = unreal.DirectoryPath(OUTPUT_FOLDER)
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
        game.use_high_quality_shadows = True
        game.shadow_distance_scale = 10
        game.shadow_radius_threshold = 0.001
        game.override_view_distance_scale = True
        game.view_distance_scale = 50

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

        global_executor = unreal.MoviePipelinePIEExecutor()

        def on_finished(executor, success):
            global global_executor
            unreal.log("=" * 60)
            unreal.log(f"RENDER COMPLETE! Success: {success}")
            unreal.log("Cleaning up gap frames and renumbering...")
            cleanup_and_renumber_frames()
            unreal.log(f"Output: {OUTPUT_FOLDER}")
            unreal.log("=" * 60)
            global_executor = None

        global_executor.on_executor_finished_delegate.add_callable(on_finished)
        mrq.render_queue_with_executor_instance(global_executor)


def cleanup_and_renumber_frames():
    """
    Post-render cleanup:
    - Delete odd-numbered gap frames (001, 003, 005...)
    - Rename even frames to sequential (000000, 000002... -> 000000, 000001...)
    """
    png_files = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "*.png")))

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
            new_path = os.path.join(OUTPUT_FOLDER, f"{new_num:06d}.png")
            if png_path != new_path:
                os.rename(png_path, new_path)
                renamed_count += 1

    unreal.log(f"  Deleted {deleted_count} gap frames")
    unreal.log(f"  Renamed {renamed_count} frames to sequential numbering")


# =============================================================================
# ENTRY POINT
# =============================================================================

if 'dope_gen_v4' in dir():
    del dope_gen_v4

if 'global_executor' in dir() and global_executor:
    global_executor = None

dope_gen_v4 = DOPEDatasetGenerator()
