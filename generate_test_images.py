"""
UE5.7.1 Test Image Generator
Captures images of tagged objects without annotations, for testing trained models.

Output Format:
    OUTPUT_FOLDER/
    ├── 000000.png
    ├── 000001.png
    └── ...

Usage:
    1. Tag objects with 'TrainObject' and camera with 'AUV_Camera' in UE5
    2. Configure OUTPUT_FOLDER, NUM_SAMPLES, and camera parameters below
    3. Execute via UE5 Python console
"""

import unreal
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
OUTPUT_FOLDER = "D:/UE5_Test_Images/"
SEQUENCE_PATH = "/Game/Generated/TestImageSequence"
NUM_SAMPLES = 100

# Camera Movement
MIN_DISTANCE = 30.0   # cm
MAX_DISTANCE = 150.0   # cm

# Resolution
RESOLUTION_X = 1920
RESOLUTION_Y = 1080

# Pool Bounds
POOL_BOUNDS = {
    "x_min": -1776.0, "x_max": 989.0,
    "y_min": -3992.0, "y_max": 690.0,
    "z_min": -1841.0, "z_max": -1360.0
}

# Camera Intrinsics
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 20.25  # Matches 16:9 aspect ratio (36 / 1.777...)
FOCAL_LENGTH_MM = 30.0

# Render Settings - Aggressive anti-ghosting
WARMUP_FRAMES = 64
SPATIAL_SAMPLES = 4
TEMPORAL_SAMPLES = 1  # CRITICAL: Keep at 1 to avoid ghosting

# Object Randomization (table-top variation)
RANDOMIZE_OBJECTS = True
OBJECT_XY_RANGE_X = 18.0    # cm — max X displacement from original position
OBJECT_XY_RANGE_Y = 15.0    # cm — max Y displacement from original position
OBJECT_YAW_RANGE = 360.0    # degrees — full spin on table
OBJECT_ROLL_RANGE = 90.0    # degrees — ±range for roll (requires pivot at base)
OBJECT_PITCH_MIN = -90.0    # degrees — min pitch
OBJECT_PITCH_MAX = 0.0      # degrees — max pitch

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


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class TestImageGenerator:
    def __init__(self):
        self.targets = []
        self.camera = None
        self.sample_data = []

        unreal.log("=" * 60)
        unreal.log("UE5.7 TEST IMAGE GENERATOR (no annotations)")
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

        unreal.log(f"Found {len(self.targets)} targets")
        unreal.log(f"Camera: {self.camera.get_actor_label()}")

        self._configure_camera()
        self._run()

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
                actor.destroy_actor()
                stale_count += 1
        if stale_count:
            unreal.log(f"  Cleaned up {stale_count} stale SceneCapture2D actor(s)")

    def _configure_camera(self):
        """Force camera settings to match render output exactly."""
        if not self.camera:
            return

        cine_comp = self.camera.get_cine_camera_component()
        cine_comp.filmback.sensor_width = SENSOR_WIDTH_MM
        cine_comp.filmback.sensor_height = SENSOR_HEIGHT_MM
        cine_comp.current_focal_length = FOCAL_LENGTH_MM
        cine_comp.focus_settings.focus_method = unreal.CameraFocusMethod.DISABLE

        unreal.log(f"  Camera Configured: Sensor {SENSOR_WIDTH_MM}x{SENSOR_HEIGHT_MM}mm "
                   f"(16:9), FL {FOCAL_LENGTH_MM}mm, Focus Breathing DISABLED")

    def _run(self):
        """Main pipeline: create sequence then render."""
        sequence = self._create_sequence()
        if not sequence:
            return

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

        frames_per_sample = 1
        total_frames = NUM_SAMPLES * frames_per_sample

        transform_section.set_range(0, total_frames + 10)
        seq.set_playback_start(0)
        seq.set_playback_end(total_frames)

        channels = transform_section.get_all_channels()

        # Create tracks for all targets to randomize their positions
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

        unreal.log("Generating camera positions...")

        for i in range(NUM_SAMPLES):
            target = random.choice(self.targets)

            # Generate randomized transforms for ALL targets this frame
            frame_target_transforms = {}
            for t in self.targets:
                orig = target_tracks[t]
                if RANDOMIZE_OBJECTS:
                    rand_loc = unreal.Vector(
                        orig['orig_loc'].x + random.uniform(-OBJECT_XY_RANGE_X, OBJECT_XY_RANGE_X),
                        orig['orig_loc'].y + random.uniform(-OBJECT_XY_RANGE_Y, OBJECT_XY_RANGE_Y),
                        orig['orig_loc'].z  # keep on table surface
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

            target_loc = frame_target_transforms[target]['loc']

            cam_pos = generate_clamped_position(target_loc)
            cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, target_loc)

            self.sample_data.append({
                "frame_idx": i,
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

        # Set CONSTANT interpolation
        for channel in channels:
            for key in channel.get_keys():
                key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)

        for track_data in target_tracks.values():
            for t_channel in track_data['channels']:
                for key in t_channel.get_keys():
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
        except Exception:
            try:
                return seq.add_track(unreal.MovieSceneCameraCutTrack)
            except Exception:
                return None

    def _render(self, sequence):
        """Execute MRQ render."""
        global global_executor

        unreal.log("Starting MRQ render...")

        mrq = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        queue = mrq.get_queue()
        queue.delete_all_jobs()

        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
        job.map = unreal.SoftObjectPath(get_world().get_path_name())
        job.job_name = "Test_Images"

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
        output.custom_end_frame = NUM_SAMPLES
        output.handle_frame_count = 0

        # Anti-ghosting settings
        aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
        aa.spatial_sample_count = SPATIAL_SAMPLES
        aa.temporal_sample_count = TEMPORAL_SAMPLES
        aa.override_anti_aliasing = True
        aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_FXAA
        aa.render_warm_up_count = WARMUP_FRAMES
        aa.engine_warm_up_count = WARMUP_FRAMES
        aa.render_warm_up_frames = True

        # High quality settings
        game = config.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
        game.cinematic_quality_settings = True
        game.texture_streaming = unreal.MoviePipelineTextureStreamingMethod.DISABLED
        game.use_lod_zero = True
        game.disable_hlods = True

        # Console commands to disable temporal effects
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

        def on_finished(executor, success):
            global global_executor
            unreal.log("=" * 60)
            unreal.log(f"RENDER COMPLETE! Success: {success}")
            renumber_frames()
            unreal.log(f"Output: {OUTPUT_FOLDER}")
            unreal.log(f"Images ready for model inference.")
            unreal.log("=" * 60)
            global_executor = None

        global_executor.on_executor_finished_delegate.add_callable(on_finished)
        mrq.render_queue_with_executor_instance(global_executor)


def renumber_frames():
    """Renumber frames sequentially and flatten any MRQ subdirectories."""
    png_files = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "**", "*.png"), recursive=True))

    renamed_count = 0
    for idx, png_path in enumerate(png_files):
        new_path = os.path.join(OUTPUT_FOLDER, f"{idx:06d}.png")
        if png_path != new_path:
            os.rename(png_path, new_path)
            renamed_count += 1

    # Remove any subdirectories MRQ may have created
    for item in os.listdir(OUTPUT_FOLDER):
        item_path = os.path.join(OUTPUT_FOLDER, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

    unreal.log(f"  Renamed {renamed_count} frames")
    unreal.log(f"  Final image count: {len(glob.glob(os.path.join(OUTPUT_FOLDER, '*.png')))}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if 'test_img_gen' in dir():
    del test_img_gen

if 'global_executor' in dir() and global_executor:
    global_executor = None

test_img_gen = TestImageGenerator()
