"""
UE5.7.1 YOLO Detection Dataset Generator
Generates synthetic training data for YOLO object detection

Output Format (per image):
    labels/{frame:06d}.txt - One line per object: class_id x_center y_center width height
    images/{frame:06d}.png - Rendered image
    classes.txt - Class name mapping

All coordinates are normalized to [0, 1] relative to image dimensions.
Reference: https://docs.ultralytics.com/datasets/detect/
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
OUTPUT_FOLDER = "D:/UE5_YOLO_Data/"
SEQUENCE_PATH = "/Game/Generated/YOLOSequence"
NUM_SAMPLES = 40

# Train/Val split ratio (fraction of data used for validation)
VAL_SPLIT_RATIO = 0.2

# Camera Movement
MIN_DISTANCE = 100.0   # cm
MAX_DISTANCE = 400.0   # cm

# Resolution
RESOLUTION_X = 1920
RESOLUTION_Y = 1080

# Pool Bounds
POOL_BOUNDS = {
    "x_min": -1776.0, "x_max": 989.0,
    "y_min": -3992.0, "y_max": 690.0,
    "z_min": -1841.0, "z_max": -1360.0
}

# Camera Intrinsics (for projection)
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 20.25  # Matches 16:9 aspect ratio (36 / 1.777...)
FOCAL_LENGTH_MM = 30.0

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

    # UE5 -> OpenCV coordinate system
    cv_x = local.y
    cv_y = -local.z
    cv_z = local.x

    if cv_z <= 0:
        return None  # Behind camera

    u = (cv_x * intrinsics["fx"] / cv_z) + intrinsics["cx"]
    v = (cv_y * intrinsics["fy"] / cv_z) + intrinsics["cy"]
    return (u, v)


def get_aabb_corners(actor):
    """Get 8 corners of the actor's axis-aligned bounding box."""
    origin, extent = actor.get_actor_bounds(False)
    ex, ey, ez = extent.x, extent.y, extent.z
    ox, oy, oz = origin.x, origin.y, origin.z

    return [
        unreal.Vector(ox + ex, oy + ey, oz + ez),
        unreal.Vector(ox + ex, oy + ey, oz - ez),
        unreal.Vector(ox + ex, oy - ey, oz + ez),
        unreal.Vector(ox + ex, oy - ey, oz - ez),
        unreal.Vector(ox - ex, oy + ey, oz + ez),
        unreal.Vector(ox - ex, oy + ey, oz - ez),
        unreal.Vector(ox - ex, oy - ey, oz + ez),
        unreal.Vector(ox - ex, oy - ey, oz - ez),
    ]


def get_2d_bbox(actor, cam_transform, intrinsics):
    """
    Get 2D axis-aligned bounding box from projected 3D corners.
    Returns (x_center, y_center, width, height) normalized to [0,1] or None if invalid.
    """
    corners_3d = get_aabb_corners(actor)

    # Project all corners
    points_2d = []
    for corner in corners_3d:
        pt = project_point(corner, cam_transform, intrinsics)
        if pt is not None:
            points_2d.append(pt)

    if len(points_2d) < 4:
        return None  # Not enough visible points

    # Get bounding box
    x_coords = [p[0] for p in points_2d]
    y_coords = [p[1] for p in points_2d]

    x_min = max(0, min(x_coords))
    x_max = min(RESOLUTION_X, max(x_coords))
    y_min = max(0, min(y_coords))
    y_max = min(RESOLUTION_Y, max(y_coords))

    # Check if box is valid
    if x_max <= x_min or y_max <= y_min:
        return None

    # Calculate normalized center and dimensions
    x_center = ((x_min + x_max) / 2.0) / RESOLUTION_X
    y_center = ((y_min + y_max) / 2.0) / RESOLUTION_Y
    width = (x_max - x_min) / RESOLUTION_X
    height = (y_max - y_min) / RESOLUTION_Y

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return (x_center, y_center, width, height)


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

class YOLODatasetGenerator:
    def __init__(self):
        self.targets = []
        self.camera = None
        self.intrinsics = calculate_intrinsics()
        self.sample_data = []
        self.class_map = {}  # actor_label -> class_id

        unreal.log("=" * 60)
        unreal.log("UE5.7 YOLO DETECTION DATASET GENERATOR")
        unreal.log("=" * 60)

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

        # Build class map
        self._build_class_map()

        unreal.log(f"Found {len(self.targets)} targets, {len(self.class_map)} classes")
        unreal.log(f"Camera: {self.camera.get_actor_label()}")

        self._configure_camera()

        # Run pipeline
        self._run()

    def _configure_camera(self):
        """Force camera settings to match render output exactly to prevent drift."""
        if not self.camera:
            return

        cine_comp = self.camera.get_cine_camera_component()

        # 1. Match Sensor Aspect Ratio to Output Resolution (16:9)
        # This prevents Unreal from applying hidden crops or FOV adjustments
        # when the sensor aspect ratio differs from render resolution.
        cine_comp.filmback.sensor_width = SENSOR_WIDTH_MM
        cine_comp.filmback.sensor_height = SENSOR_HEIGHT_MM

        # 2. Lock Focal Length & Disable Focus Breathing
        # Focus breathing changes FOV based on focus distance, causing labels to drift.
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

        # Generate labels before rendering
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

        for i in range(NUM_SAMPLES):
            target = random.choice(self.targets)
            target_loc = target.get_actor_location()

            cam_pos = generate_clamped_position(target_loc)
            cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_pos, target_loc)

            self.sample_data.append({
                "frame_idx": i,
                "target": target,
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

        # Set CONSTANT interpolation
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
        except Exception:
            try:
                return seq.add_track(unreal.MovieSceneCameraCutTrack)
            except Exception:
                return None

    def _generate_all_labels(self):
        """Generate YOLO format label files.

        For each frame, ALL target actors are tested (not just the one
        used for camera aiming), so multiple objects can appear per label.
        """
        unreal.log("Generating YOLO label files...")

        total_boxes = 0
        empty_frames = 0

        with unreal.ScopedSlowTask(NUM_SAMPLES, "Generating YOLO Labels...") as slow_task:
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

                # Try ALL targets from this viewpoint
                label_lines = []
                for target in self.targets:
                    bbox = get_2d_bbox(target, cam_transform, self.intrinsics)
                    if bbox:
                        class_id = self.class_map[target.get_actor_label()]
                        x_center, y_center, width, height = bbox
                        label_lines.append(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )
                        total_boxes += 1

                # Write label file (may have 0, 1, or multiple objects)
                label_path = os.path.join(self.staging_labels, f"{i:06d}.txt")
                with open(label_path, 'w') as f:
                    f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

                if not label_lines:
                    empty_frames += 1

                if (i + 1) % 10 == 0:
                    unreal.log(f"  Progress: {i + 1}/{NUM_SAMPLES} frames")

        unreal.log(f"  Labels complete: {total_boxes} boxes across "
                   f"{NUM_SAMPLES} frames ({empty_frames} empty frames)")

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
        output.custom_end_frame = NUM_SAMPLES * 2
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
            unreal.log("Dataset is ready for: yolo detect train data=data.yaml")
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

if 'yolo_gen' in dir():
    del yolo_gen

if 'global_executor' in dir() and global_executor:
    global_executor = None

yolo_gen = YOLODatasetGenerator()
