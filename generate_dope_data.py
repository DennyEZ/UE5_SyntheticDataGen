"""
UE5.7.1 DOPE Dataset Generator - V3 (Shot-Based Anti-Ghosting)
Generates synthetic training data for Deep Object Pose Estimation (DOPE)

This version uses a SHOT-BASED approach:
- Each camera position is a separate SHOT in the sequence
- MRQ resets temporal state between shots automatically
- Single MRQ job execution (more efficient than V2)
- Uses FXAA instead of TAA to eliminate temporal artifacts

Key improvements:
- Aggressive anti-ghosting via console variables
- DOPE-compatible JSON format with per-image files
- Proper coordinate system transformation (UE5 -> OpenCV)
- Correct cuboid corner ordering matching DOPE/NDDS standard
"""

import unreal
import json
import math
import os
import shutil
import random
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

# Scene Tags
TARGET_TAG = "TrainObject"
CAMERA_TAG = "AUV_Camera"

# Output Settings
OUTPUT_FOLDER = "D:/UE5_Data/"
SEQUENCE_PATH = "/Game/Generated/DOPESequence"
NUM_SAMPLES = 40

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
    # Get camera inverse transform (this correctly inverts the rotation)
    cam_inv = cam_transform.inverse()
    cam_rot_inv = cam_inv.rotation

    # Convert obj_rot (Rotator) to Quat
    obj_quat = obj_rot.quaternion()

    # Combine rotations using Quat multiplication operator
    # relative = cam_inv_rotation * obj_rotation
    relative_quat = cam_rot_inv * obj_quat

    # Normalize
    length = math.sqrt(relative_quat.x**2 + relative_quat.y**2 +
                       relative_quat.z**2 + relative_quat.w**2)
    if length > 0:
        qx = relative_quat.x / length
        qy = relative_quat.y / length
        qz = relative_quat.z / length
        qw = relative_quat.w / length
    else:
        qx, qy, qz, qw = 0, 0, 0, 1

    # Coordinate system transformation (UE5 -> OpenCV)
    # UE5: X-forward, Y-right, Z-up
    # OpenCV: X-right, Y-down, Z-forward
    return [qy, -qz, qx, qw]


def get_cuboid_corners(actor):
    """Get 9 cuboid points (8 corners + centroid) in DOPE order."""
    origin, extent = actor.get_actor_bounds(False)
    ex, ey, ez = extent.x, extent.y, extent.z
    ox, oy, oz = origin.x, origin.y, origin.z

    # Raw corners
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

    # UE5 -> OpenCV
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
# MAIN GENERATOR CLASS
# =============================================================================

class DOPEDatasetGenerator:
    def __init__(self):
        self.targets = []
        self.camera = None
        self.intrinsics = calculate_intrinsics()
        self.sample_data = []  # Store data for JSON generation

        unreal.log("=" * 60)
        unreal.log("UE5.7 DOPE DATASET GENERATOR V3 (Shot-Based)")
        unreal.log("=" * 60)

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

        self._configure_camera()

        # Run pipeline
        self._run()

    def _configure_camera(self):
        """Force camera settings to match render output exactly to prevent drift."""
        if not self.camera:
            return

        cine_comp = self.camera.get_cine_camera_component()

        # 1. Match Sensor Aspect Ratio to Output Resolution (16:9)
        cine_comp.filmback.sensor_width = SENSOR_WIDTH_MM
        cine_comp.filmback.sensor_height = SENSOR_HEIGHT_MM

        # 2. Lock Focal Length & Disable Focus Breathing
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

    def _run(self):
        """Main pipeline."""
        # Create sequence with one shot per sample
        sequence = self._create_sequence()
        if not sequence:
            return

        # Generate all JSON files now (before rendering)
        self._generate_all_jsons()

        # Save and render
        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
        self._render(sequence)

    def _create_sequence(self):
        """Create level sequence with multiple shots (one per sample)."""
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

        if unreal.EditorAssetLibrary.does_asset_exist(SEQUENCE_PATH):
            unreal.EditorAssetLibrary.delete_asset(SEQUENCE_PATH)

        pkg_path, asset_name = SEQUENCE_PATH.rsplit('/', 1)
        seq = asset_tools.create_asset(
            asset_name, pkg_path, unreal.LevelSequence, unreal.LevelSequenceFactoryNew()
        )

        # Frame rate
        seq.set_display_rate(unreal.FrameRate(24, 1))

        # Add camera binding
        cam_binding = seq.add_possessable(self.camera)
        transform_track = cam_binding.add_track(unreal.MovieScene3DTransformTrack)
        transform_section = transform_track.add_section()

        # Each sample gets 2 frames (1 frame of content + 1 frame gap)
        # This gives MRQ time to reset between samples
        frames_per_sample = 2
        total_frames = NUM_SAMPLES * frames_per_sample

        transform_section.set_range(0, total_frames + 10)
        seq.set_playback_start(0)
        seq.set_playback_end(total_frames)

        channels = transform_section.get_all_channels()

        unreal.log("Generating camera positions...")

        for i in range(NUM_SAMPLES):
            target = random.choice(self.targets)
            target_loc = target.get_actor_location()
            target_rot = target.get_actor_rotation()

            cam_pos = generate_clamped_position(target_loc)
            cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, target_loc)

            # Store for JSON generation
            self.sample_data.append({
                "frame_idx": i,
                "target": target,
                "cam_pos": cam_pos,
                "cam_rot": cam_rot,
                "target_loc": target_loc,
                "target_rot": target_rot
            })

            # Add keyframe at frame i * frames_per_sample
            frame_num = i * frames_per_sample
            frame_time = unreal.FrameNumber(frame_num)

            # Position
            channels[0].add_key(frame_time, cam_pos.x)
            channels[1].add_key(frame_time, cam_pos.y)
            channels[2].add_key(frame_time, cam_pos.z)

            # Rotation
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
        """Generate all JSON files for the dataset.

        For each frame, ALL target actors are annotated (not just the one
        used for camera aiming), so multiple objects appear per JSON.
        """
        unreal.log("Generating JSON files...")

        total_objects = 0

        with unreal.ScopedSlowTask(NUM_SAMPLES, "Generating DOPE JSON Files...") as slow_task:
            slow_task.make_dialog(True)  # Show dialog with cancel button

            for data in self.sample_data:
                # Check if user cancelled
                if slow_task.should_cancel():
                    break

                slow_task.enter_progress_frame(1)

                i = data["frame_idx"]
                cam_pos = data["cam_pos"]
                cam_rot = data["cam_rot"]

                cam_transform = unreal.Transform(location=cam_pos, rotation=cam_rot)

                # Annotate ALL targets from this viewpoint
                objects_list = []
                for target in self.targets:
                    target_loc = target.get_actor_location()
                    target_rot = target.get_actor_rotation()

                    # Project cuboid
                    corners = get_cuboid_corners(target)
                    projected = [project_point(c, cam_transform, self.intrinsics) for c in corners]

                    # Check if the centroid (last point) is behind the camera
                    if projected[-1] == [-9999.0, -9999.0]:
                        continue

                    objects_list.append({
                        "class": target.get_actor_label(),
                        "name": f"{target.get_actor_label()}_{i:03d}",
                        "visibility": 1.0,
                        "location": ue_to_opencv_location(target_loc, cam_transform),
                        "quaternion_xyzw": ue_rotation_to_quaternion_xyzw(target_rot, cam_transform),
                        "projected_cuboid": projected
                    })
                    total_objects += 1

                # DOPE JSON format
                json_data = {
                    "camera_data": {
                        "width": RESOLUTION_X,
                        "height": RESOLUTION_Y,
                        "intrinsics": self.intrinsics
                    },
                    "objects": objects_list
                }

                # Save with sequential numbering (final output format)
                json_path = os.path.join(OUTPUT_FOLDER, f"{i:06d}.json")
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=4)

                if (i + 1) % 10 == 0:
                    unreal.log(f"  Progress: {i + 1}/{NUM_SAMPLES} JSON files")

        unreal.log(f"  Generated {len(self.sample_data)} JSON files "
                   f"with {total_objects} total object annotations")

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

        # PNG output
        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)

        # Output settings
        output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        output.output_directory = unreal.DirectoryPath(OUTPUT_FOLDER)
        output.output_resolution = unreal.IntPoint(RESOLUTION_X, RESOLUTION_Y)
        output.file_name_format = "{frame_number}"
        output.zero_pad_frame_numbers = 6
        output.flush_disk_writes_per_shot = True
        output.use_custom_playback_range = True
        output.custom_start_frame = 0
        output.custom_end_frame = NUM_SAMPLES * 2  # Every other frame
        output.handle_frame_count = 0

        # CRITICAL: Anti-ghosting AA settings
        aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
        aa.spatial_sample_count = SPATIAL_SAMPLES
        aa.temporal_sample_count = TEMPORAL_SAMPLES  # NO TEMPORAL ACCUMULATION
        aa.override_anti_aliasing = True
        aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_FXAA  # Non-temporal AA
        aa.render_warm_up_count = WARMUP_FRAMES
        aa.engine_warm_up_count = WARMUP_FRAMES
        aa.render_warm_up_frames = True

        # High quality overrides
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

        # CRITICAL: Console commands to disable ALL temporal effects
        console = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
        console.start_console_commands = [
            # Disable TAA completely
            "r.TemporalAA 0",
            "r.TemporalAA.Quality 0",
            "r.TemporalAACurrentFrameWeight 1.0",
            "r.TemporalAASamples 1",
            "r.TemporalAAFilterSize 0",

            # Disable TSR
            "r.TSR.History.ScreenPercentage 100",
            "r.TSR.History.UpdatePersistentFeedback 0",
            "r.TSR.ShadingRejection.Flickering 0",

            # Disable motion blur
            "r.MotionBlurQuality 0",
            "r.MotionBlur.Max 0",
            "r.DefaultFeature.MotionBlur 0",

            # Disable temporal SSR/DOF
            "r.SSR.Temporal 0",
            "r.DOF.TemporalAAQuality 0",

            # === DISABLE DEPTH OF FIELD (FIX BLUR) ===
            "r.DepthOfFieldQuality 0",
            "r.DOF.Kernel.MaxForegroundRadius 0",
            "r.DOF.Kernel.MaxBackgroundRadius 0",
            "r.DepthOfField.MaxSize 0",
            "ShowFlag.DepthOfField 0",

            # Force FXAA
            "r.DefaultFeature.AntiAliasing 1",

            # Volumetric temporal off
            "r.VolumetricFog.TemporalReprojection 0",

            # Screen percentage
            "r.ScreenPercentage 100",
        ]

        # Deferred pass
        config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)

        # Create executor with callback
        global_executor = unreal.MoviePipelinePIEExecutor()

        def on_finished(executor, success):
            global global_executor
            unreal.log("=" * 60)
            unreal.log(f"RENDER COMPLETE! Success: {success}")

            # Post-render cleanup: rename gap frames and clean up
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
    # Find all PNG files
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
            # Odd frame = gap frame, delete it
            os.remove(png_path)
            deleted_count += 1
        else:
            # Even frame, rename to sequential
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

if 'dope_gen_v3' in dir():
    del dope_gen_v3

if 'global_executor' in dir() and global_executor:
    global_executor = None

dope_gen_v3 = DOPEDatasetGenerator()
