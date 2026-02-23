"""
UE5.7.1 YOLO Segmentation Dataset Generator (SceneCapture Mask)
Generates synthetic training data for YOLO instance segmentation

Output Format (per image):
    labels/{frame:06d}.txt - One line per object: class_id x1 y1 x2 y2 x3 y3 ... (polygon vertices)
    images/{frame:06d}.png - Rendered image
    classes.txt - Class name mapping

All coordinates are normalized to [0, 1] relative to image dimensions.

Segmentation approach (SceneCapture2D two-pass differential):
    Inspired by CARLA simulator's instance segmentation camera.
    For each camera viewpoint and target actor, a SceneCapture2D takes two captures:
      Pass 1 (background): Full scene rendered WITH the target hidden.
      Pass 2 (foreground): Full scene rendered WITH the target visible.
    The absolute difference |foreground - background| produces a mask of ONLY the
    visible portion of the target, with natural occlusion from scene geometry
    (e.g. tables, walls) automatically handled.

    This produces pixel-accurate, occlusion-aware silhouettes for any mesh type
    (including Nanite meshes) without requiring mesh vertex extraction or custom
    materials. FOV is read directly from the CineCameraComponent for exact alignment.

Prerequisites:
    Install opencv in UE5's Python:
        <UE5_ROOT>/Engine/Binaries/ThirdParty/Python3/Win64/python.exe -m pip install opencv-python-headless

Reference: https://docs.ultralytics.com/datasets/segment/
"""

import unreal
import math
import os
import shutil
import random
import glob

# Try to import cv2 for contour extraction
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
OUTPUT_FOLDER = "D:/UE5_YOLO_Seg_Data/"
SEQUENCE_PATH = "/Game/Generated/YOLOSegSequence"
NUM_SAMPLES = 40

# Train/Val split ratio (fraction of data used for validation)
VAL_SPLIT_RATIO = 0.2

# Fraction of images that will be negative samples (no objects)
NEGATIVE_SAMPLE_RATIO = 0.1

# Camera jitter (random tilt to avoid center bias)
CAM_JITTER_MAX_PITCH = 15.0    # degrees, ±range for pitch offset
CAM_JITTER_MAX_YAW = 15.0      # degrees, ±range for yaw offset

# Camera Movement
MIN_DISTANCE = 100.0   # cm
MAX_DISTANCE = 400.0   # cm

# Render resolution (final images)
RESOLUTION_X = 1280
RESOLUTION_Y = 720

# Mask capture resolution
# IMPORTANT: Must match render resolution exactly to avoid FOV/alignment drift.
# SceneCapture2D and CineCamera compute FOV differently, and at different resolutions
# subtle pixel-rounding differences cause the mask to shift relative to final render.
MASK_RESOLUTION_X = RESOLUTION_X
MASK_RESOLUTION_Y = RESOLUTION_Y

# Pool Bounds
POOL_BOUNDS = {
    "x_min": -1776.0, "x_max": 989.0,
    "y_min": -3992.0, "y_max": 690.0,
    "z_min": -1841.0, "z_max": -1360.0
}

# Camera Intrinsics (for SceneCapture FOV matching)
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 20.25  # Matches 16:9 aspect ratio of 1920x1080 (36 / 1.777...)
FOCAL_LENGTH_MM = 30.0

# Render Settings
WARMUP_FRAMES = 16
SPATIAL_SAMPLES = 1
TEMPORAL_SAMPLES = 1

# Polygon simplification factor (fraction of contour arc length)
# Lower = more vertices = more accurate outline
# Higher = fewer vertices = faster training
POLYGON_EPSILON_FACTOR = 0.002

# Minimum contour area in pixels (at mask resolution) to be considered valid
# Scaled down from full-res: 100 * (480*270) / (1920*1080) ≈ 6
MIN_CONTOUR_AREA = 6

# Save debug mask images (bg, fg, diff, binary) for frame 0
# Useful for diagnosing mask alignment; disable for clean dataset output
SAVE_DEBUG_MASKS = False

# Global reference to prevent garbage collection
global_executor = None

# Render target asset path (created as a UE asset in content browser)
RT_ASSET_PATH = "/Game/Generated/SegMaskRT"

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
    """Calculate camera intrinsic parameters."""
    px_per_mm_x = RESOLUTION_X / SENSOR_WIDTH_MM
    px_per_mm_y = RESOLUTION_Y / SENSOR_HEIGHT_MM

    fx = FOCAL_LENGTH_MM * px_per_mm_x
    fy = FOCAL_LENGTH_MM * px_per_mm_y
    cx = RESOLUTION_X / 2.0
    cy = RESOLUTION_Y / 2.0

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


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
# SCENE CAPTURE MASK EXTRACTION
# =============================================================================

def create_render_target_asset(width, height):

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
    Spawn a SceneCapture2D actor and render target for per-actor mask capture.

    Uses PRM_RENDER_SCENE_NORMALLY so the full scene (including occluding
    geometry like tables, walls) is rendered. We then use a two-pass
    differential approach to extract only the visible portion of each target.

    Reads the actual FOV from the CineCameraComponent to avoid mismatch.

    Args:
        camera_actor: The CineCameraActor to match FOV from

    Returns:
        (capture_actor, render_target) or (None, None) on failure
    """
    # Create render target at mask resolution (lower res = much faster pixel reads)
    rt = create_render_target_asset(MASK_RESOLUTION_X, MASK_RESOLUTION_Y)

    if not rt:
        unreal.log_error("Failed to create render target!")
        return None, None

    # Read the actual FOV from the CineCameraComponent
    try:
        cine_comp = camera_actor.get_cine_camera_component()
        fov_degrees = cine_comp.field_of_view
    except (AttributeError, Exception):
        # Fallback: compute from intrinsics
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

    # Configure the capture component
    cc = capture_actor.capture_component2d
    cc.texture_target = rt

    # CRITICAL: Render the FULL scene (not show-only) so environment geometry
    # naturally occludes target objects (e.g. table edges, walls).
    # We use a two-pass differential approach to isolate each target.
    cc.set_editor_property('primitive_render_mode',
                           unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES)

    # Use BaseColor capture: renders material albedo without lighting artifacts.
    # This ensures consistent colors between the two passes for reliable diffing.
    cc.set_editor_property('capture_source',
                           unreal.SceneCaptureSource.SCS_BASE_COLOR)

    # Match the main camera FOV (read from the actual CineCamera)
    cc.set_editor_property('fov_angle', fov_degrees)

    # Manual capture only (we trigger via capture_scene())
    cc.set_editor_property('capture_every_frame', False)
    cc.set_editor_property('capture_on_movement', False)

    unreal.log(f"  SceneCapture2D ready (FOV: {fov_degrees:.1f} deg from CineCamera, "
               f"mask {MASK_RESOLUTION_X}x{MASK_RESOLUTION_Y}, full-scene differential mode)")

    return capture_actor, rt


def _read_render_target_as_numpy(render_target):
    """
    Read the render target contents into a numpy array (H, W, 3) uint8 BGR.
    Uses RenderingLibrary.read_render_target() for bulk pixel read.
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


def capture_actor_mask(capture_actor, render_target, cine_camera, cam_pos, cam_rot,
                       target_actor, frame_idx):
    """
    Capture a visibility-aware mask of target_actor using two-pass differential.

    Two-pass approach:
      Pass 1 (background): Hide the target actor, capture full scene.
      Pass 2 (foreground): Show the target actor, capture full scene.
      Diff: |foreground - background| gives only the visible pixels of the
            target, with natural occlusion from scene geometry.

    This handles:
      - Objects partially behind tables/walls (only visible portion is masked)
      - Exact FOV alignment (same camera renders both passes)

    Args:
        capture_actor: The SceneCapture2D actor
        render_target: The TextureRenderTarget2D
        cine_camera: The CineCameraActor (moved to position before capture)
        cam_pos: Camera position (unreal.Vector)
        cam_rot: Camera rotation (unreal.Rotator)
        target_actor: The actor to create a mask for
        frame_idx: Frame index (for logging)

    Returns:
        list of (x, y) normalized polygon points, or None
    """
    cc = capture_actor.capture_component2d

    # CRITICAL: Move the actual CineCamera to this position first.
    # This ensures the CineCameraComponent's world transform matches MRQ render.
    cine_camera.set_actor_location_and_rotation(
        cam_pos, cam_rot, False, True  # sweep=False, teleport=True
    )

    # Now read the CineCameraComponent's actual world transform
    # (which may differ from actor root due to component offset)
    try:
        cine_comp = cine_camera.get_cine_camera_component()
        actual_pos = cine_comp.get_world_location()
        actual_rot = cine_comp.get_world_rotation()
    except (AttributeError, Exception):
        # Fallback to actor transform if component access fails
        actual_pos = cam_pos
        actual_rot = cam_rot

    # Position the SceneCapture at the CineCameraComponent's exact world transform
    cc.set_world_location_and_rotation(
        actual_pos, actual_rot, False, True  # sweep=False, teleport=True
    )


    # --- Pass 1: Background (target hidden) ---
    target_actor.set_actor_hidden_in_game(True)
    cc.capture_scene()
    bg = _read_render_target_as_numpy(render_target)

    # --- Pass 2: Foreground (target visible) ---
    target_actor.set_actor_hidden_in_game(False)
    cc.capture_scene()
    fg = _read_render_target_as_numpy(render_target)

    if bg is None or fg is None:
        return None

    # --- Compute difference mask ---
    diff = cv2.absdiff(fg, bg)
    diff_gray = np.max(diff, axis=2)

    # Threshold: any pixel that changed by more than a small amount = target
    _, binary = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)

    # Morphological cleanup to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # REMOVED Morph Open (erosion) to preserve thin structures like ladle handles
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Save debug images for first frame to help diagnose alignment
    if SAVE_DEBUG_MASKS and frame_idx == 0:
        debug_dir = os.path.join(OUTPUT_FOLDER, "_debug_masks")
        os.makedirs(debug_dir, exist_ok=True)
        actor_name = target_actor.get_actor_label()
        cv2.imwrite(os.path.join(debug_dir, f"{actor_name}_bg.png"), bg)
        cv2.imwrite(os.path.join(debug_dir, f"{actor_name}_fg.png"), fg)
        cv2.imwrite(os.path.join(debug_dir, f"{actor_name}_diff.png"), diff)
        cv2.imwrite(os.path.join(debug_dir, f"{actor_name}_binary.png"), binary)
        unreal.log(f"  Debug masks saved for '{actor_name}' frame 0 → {debug_dir}")

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    polygons = []
    h, w = fg.shape[:2]

    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue

        # Simplify to reduce vertex count while preserving shape
        epsilon = POLYGON_EPSILON_FACTOR * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # Normalize coordinates to [0, 1]
        poly_points = []
        for point in approx:
            px, py = point[0]
            poly_points.append((
                max(0.0, min(1.0, float(px) / w)),
                max(0.0, min(1.0, float(py) / h))
            ))
        polygons.append(poly_points)

    return polygons



# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class YOLOSegDatasetGenerator:
    def __init__(self):
        self.targets = []
        self.camera = None
        self.intrinsics = calculate_intrinsics()
        self.sample_data = []
        self.class_map = {}
        self.capture_actor = None
        self.render_target = None

        unreal.log("=" * 60)
        unreal.log("UE5.7 YOLO SEGMENTATION DATASET GENERATOR")
        unreal.log("  Mode: SceneCapture2D two-pass differential (occlusion-aware)")
        unreal.log("=" * 60)

        if not HAS_CV2:
            unreal.log_error(
                "ERROR: cv2 is required for segmentation! Install via:\n"
                "  <UE5>/Engine/Binaries/ThirdParty/Python3/Win64/python.exe "
                "-m pip install opencv-python-headless"
            )
            return

        # Setup output folders
        # Use a staging area for generation, then split into train/val later
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
        self.staging_images = os.path.join(OUTPUT_FOLDER, "_staging", "images")
        self.staging_labels = os.path.join(OUTPUT_FOLDER, "_staging", "labels")
        os.makedirs(self.staging_images)
        os.makedirs(self.staging_labels)

        # Find actors
        self._find_actors()
        if not self.camera or not self.targets:
            unreal.log_error("ERROR: Camera or targets not found!")
            return

        self._build_class_map()

        unreal.log(f"Found {len(self.targets)} targets, {len(self.class_map)} classes")
        unreal.log(f"Camera: {self.camera.get_actor_label()}")

        self._configure_camera()
        self._run()

    def _configure_camera(self):
        """Force camera settings to match render output exactly to prevent drift."""
        if not self.camera:
            return

        cine_comp = self.camera.get_cine_camera_component()

        # 1. Match Sensor Aspect Ratio to Output Resolution (16:9)
        # This prevents Unreal from applying hidden crops or FOV adjustments
        # when the sensor aspect ratio (was 3:2) differs from render (16:9).
        cine_comp.filmback.sensor_width = SENSOR_WIDTH_MM
        cine_comp.filmback.sensor_height = SENSOR_HEIGHT_MM

        # 2. Lock Focal Length & Disable Focus Breathing
        # Focus breathing changes FOV based on focus distance, causing masks to slide.
        cine_comp.current_focal_length = FOCAL_LENGTH_MM
        cine_comp.focus_settings.focus_method = unreal.CameraFocusMethod.DISABLE

        unreal.log(f"  Camera Configured: Sensor {SENSOR_WIDTH_MM}x{SENSOR_HEIGHT_MM}mm "
                   f"(16:9), FL {FOCAL_LENGTH_MM}mm, Focus Breathing DISABLED")

    def _find_actors(self):
        """Find tagged actors and clean up stale SceneCapture2D actors from previous runs."""
        subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        stale_count = 0
        for actor in subsys.get_all_level_actors():
            if actor.actor_has_tag(TARGET_TAG):
                self.targets.append(actor)
            elif actor.actor_has_tag(CAMERA_TAG):
                self.camera = actor
            elif actor.get_class().get_name() == "SceneCapture2D":
                # Remove leftover SceneCapture2D actors from crashed runs
                actor.destroy_actor()
                stale_count += 1
        if stale_count:
            unreal.log(f"  Cleaned up {stale_count} stale SceneCapture2D actor(s)")

    def _build_class_map(self):
        """Build mapping from actor labels to class IDs."""
        labels = sorted(set(t.get_actor_label() for t in self.targets))
        self.class_map = {label: idx for idx, label in enumerate(labels)}
        unreal.log(f"Class map: {self.class_map}")

    def _run(self):
        """Main pipeline."""
        sequence = self._create_sequence()
        if not sequence:
            return

        # Generate labels using SceneCapture masks
        self._generate_all_labels()

        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
        self._render(sequence)

    def _create_sequence(self):
        """Create level sequence with camera keyframes."""
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
        total_frames = NUM_SAMPLES * frames_per_sample

        transform_section.set_range(0, total_frames + 10)
        seq.set_playback_start(0)
        seq.set_playback_end(total_frames)

        channels = transform_section.get_all_channels()

        unreal.log("Generating camera positions...")

        # Setup transform tracks for all targets so we can teleport them out on negative samples
        target_tracks = []
        for target in self.targets:
            binding = seq.add_possessable(target)
            track = binding.add_track(unreal.MovieScene3DTransformTrack)
            section = track.add_section()
            section.set_range(0, total_frames + 10)
            target_tracks.append({
                'target': target,
                'channels': section.get_all_channels(),
                'orig_loc': target.get_actor_location(),
                'orig_rot': target.get_actor_rotation()
            })

        # Add initial keys for all channels at frame 0 so objects have a baseline transform
        frame_0 = unreal.FrameNumber(0)
        for t_data in target_tracks:
            c = t_data['channels']
            orig_loc = t_data['orig_loc']
            orig_rot = t_data['orig_rot']
            c[0].add_key(frame_0, orig_loc.x)
            c[1].add_key(frame_0, orig_loc.y)
            c[2].add_key(frame_0, orig_loc.z)
            c[3].add_key(frame_0, orig_rot.roll)
            c[4].add_key(frame_0, orig_rot.pitch)
            c[5].add_key(frame_0, orig_rot.yaw)

        last_is_negative = False

        for i in range(NUM_SAMPLES):
            is_negative = random.random() < NEGATIVE_SAMPLE_RATIO

            target = random.choice(self.targets)
            target_loc = target.get_actor_location()

            cam_pos = generate_clamped_position(target_loc)
            
            # Camera jitter via look-at-point offset (avoids gimbal lock)
            dist = math.sqrt((cam_pos.x - target_loc.x)**2 +
                             (cam_pos.y - target_loc.y)**2 +
                             (cam_pos.z - target_loc.z)**2)
            max_offset = dist * math.tan(math.radians(CAM_JITTER_MAX_PITCH))

            margin = 0.10  # 10% margin from edges
            cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, target_loc)  # default: centered
            jitter_scale = 1.0

            for _attempt in range(4):
                offset = max_offset * jitter_scale
                look_at_point = unreal.Vector(
                    target_loc.x + random.uniform(-offset, offset),
                    target_loc.y + random.uniform(-offset, offset),
                    target_loc.z + random.uniform(-offset * 0.5, offset * 0.5)  # less vertical
                )
                test_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, look_at_point)

                # Validate: aimed target stays in-frame
                test_transform = unreal.Transform(location=cam_pos, rotation=test_rot)
                centroid_2d = project_point(target_loc, test_transform, self.intrinsics)
                if (centroid_2d != [-9999.0, -9999.0] and
                    RESOLUTION_X * margin < centroid_2d[0] < RESOLUTION_X * (1 - margin) and
                    RESOLUTION_Y * margin < centroid_2d[1] < RESOLUTION_Y * (1 - margin)):
                    cam_rot = test_rot
                    break
                # Too much jitter — halve and retry
                jitter_scale *= 0.5

            self.sample_data.append({
                "frame_idx": i,
                "is_negative": is_negative,
                "cam_pos": cam_pos,
                "cam_rot": cam_rot,
            })

            frame_num = i * frames_per_sample
            frame_time = unreal.FrameNumber(frame_num)

            channels[0].add_key(frame_time, cam_pos.x)
            channels[1].add_key(frame_time, cam_pos.y)
            channels[2].add_key(frame_time, cam_pos.z)
            channels[3].add_key(frame_time, cam_rot.roll)
            channels[4].add_key(frame_time, cam_rot.pitch)
            channels[5].add_key(frame_time, cam_rot.yaw)

            # Optimisation: Only add keys to the Z channel, and ONLY if the state changed.
            # Writing 6 keys per target per frame is extremely slow in Unreal's Python API.
            if is_negative != last_is_negative:
                for t_data in target_tracks:
                    c = t_data['channels']
                    orig_loc = t_data['orig_loc']
                    z_val = 10000.0 if is_negative else orig_loc.z
                    
                    c[2].add_key(frame_time, z_val)
                
                last_is_negative = is_negative

        for channel in channels:
            for key in channel.get_keys():
                key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)

        for t_data in target_tracks:
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
        """Safely add camera cut track."""
        movie_scene = seq.get_movie_scene()
        try:
            return movie_scene.add_camera_cut_track()
        except Exception:
            try:
                return seq.add_track(unreal.MovieSceneCameraCutTrack)
            except Exception:
                return None

    def _generate_all_labels(self):
        """Generate YOLO segmentation labels using SceneCapture2D masks.

        Uses a two-pass differential approach per target:
          - Pass 1: scene WITHOUT target (background)
          - Pass 2: scene WITH target (foreground)
          - Diff → visible-only mask with natural occlusion

        For each frame, ALL target actors are tested (not just the one
        used for camera aiming), so multiple objects can appear per label.
        """
        unreal.log("Generating segmentation masks via SceneCapture2D...")
        unreal.log("  Mode: two-pass differential (occlusion-aware)")

        # Setup the scene capture system (reads FOV from actual CineCamera)
        self.capture_actor, self.render_target = setup_scene_capture(self.camera)
        if not self.capture_actor:
            unreal.log_error("SceneCapture2D setup failed!")
            return

        total_polys = 0
        empty_frames = 0

        # Add progress bar to prevent "freeze" perception
        with unreal.ScopedSlowTask(NUM_SAMPLES, "Generating Segmentation Masks...") as slow_task:
            slow_task.make_dialog(True)  # Show dialog with cancel button

            for data in self.sample_data:
                # Check if user cancelled
                if slow_task.should_cancel():
                    break

                slow_task.enter_progress_frame(1)
                
                i = data["frame_idx"]
                cam_pos = data["cam_pos"]
                cam_rot = data["cam_rot"]
                is_negative = data.get("is_negative", False)

                if is_negative:
                    # Skip rendering masks for negative samples, write empty label
                    label_path = os.path.join(self.staging_labels, f"{i:06d}.txt")
                    with open(label_path, 'w') as f:
                        f.write("")
                    empty_frames += 1
                    continue

                # Try ALL targets from this viewpoint
                label_lines = []
                for target in self.targets:
                    polygons = capture_actor_mask(
                        self.capture_actor, self.render_target,
                        self.camera, cam_pos, cam_rot, target, i
                    )
                    if polygons:
                        class_id = self.class_map[target.get_actor_label()]
                        for poly in polygons:
                            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                            label_lines.append(f"{class_id} {coords}")
                            total_polys += 1

                # Write label file (may have 0, 1, or multiple objects)
                label_path = os.path.join(self.staging_labels, f"{i:06d}.txt")
                with open(label_path, 'w') as f:
                    f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

                if not label_lines:
                    empty_frames += 1

                if (i + 1) % 10 == 0:
                    unreal.log(f"  Progress: {i + 1}/{NUM_SAMPLES} frames captured")

        # Cleanup: destroy the temporary capture actor
        if self.capture_actor:
            self.capture_actor.destroy_actor()
            self.capture_actor = None

        # Cleanup: delete the temporary render target asset
        if unreal.EditorAssetLibrary.does_asset_exist(RT_ASSET_PATH):
            unreal.EditorAssetLibrary.delete_asset(RT_ASSET_PATH)

        unreal.log(f"  Masks complete: {total_polys} polygons across "
                   f"{NUM_SAMPLES} frames ({empty_frames} empty frames)")

    def _render(self, sequence):
        """Execute MRQ render for RGB images."""
        global global_executor

        unreal.log("Starting MRQ render...")

        mrq = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        queue = mrq.get_queue()
        queue.delete_all_jobs()

        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
        job.map = unreal.SoftObjectPath(get_world().get_path_name())
        job.job_name = "YOLO_Seg_Dataset"

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
        output.custom_end_frame = NUM_SAMPLES * 2
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

        class_map = self.class_map  # capture for closure

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
            unreal.log("Dataset is ready for: yolo segment train data=data.yaml")
            unreal.log("=" * 60)
            global_executor = None

        global_executor.on_executor_finished_delegate.add_callable(on_finished)
        mrq.render_queue_with_executor_instance(global_executor)


def cleanup_and_renumber_frames():
    """Remove gap frames and renumber sequentially."""
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

    # Remove any subdirectories MRQ may have created
    for item in os.listdir(images_folder):
        item_path = os.path.join(images_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

    unreal.log(f"  Deleted {deleted_count} gap frames, renamed {renamed_count} frames")


def split_dataset(output_folder, val_ratio=0.2):
    """
    Split staging images/labels into train/ and val/ directories.

    Final structure:
        output_folder/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        └── val/
            ├── images/
            └── labels/
    """
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
            # Move image
            shutil.move(img_path, os.path.join(split_img_dir, f"{basename}.png"))
            # Move matching label
            lbl_path = os.path.join(staging_labels, f"{basename}.txt")
            if os.path.exists(lbl_path):
                shutil.move(lbl_path, os.path.join(split_lbl_dir, f"{basename}.txt"))

        unreal.log(f"  {split_name}: {len(img_list)} samples")

    # Remove staging directory
    staging_dir = os.path.join(output_folder, "_staging")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)


def generate_data_yaml(output_folder, class_map):
    """
    Generate data.yaml for YOLO training.

    Format:
        path: <absolute dataset root>
        train: train/images
        val: val/images
        names:
          0: class_name_0
          1: class_name_1
    """
    sorted_classes = sorted(class_map.items(), key=lambda x: x[1])
    names = {idx: name for name, idx in sorted_classes}

    # Write YAML manually to avoid PyYAML dependency
    yaml_path = os.path.join(output_folder, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_folder.rstrip('/').rstrip(chr(92))}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for idx in sorted(names.keys()):
            f.write(f"  {idx}: {names[idx]}\n")

    unreal.log(f"  data.yaml saved to {yaml_path}")
    unreal.log(f"  Classes ({len(names)}): {names}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if 'yolo_seg_gen' in dir():
    del yolo_seg_gen

if 'global_executor' in dir() and global_executor:
    global_executor = None

yolo_seg_gen = YOLOSegDatasetGenerator()
