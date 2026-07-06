# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

UE5 Synthetic Data Generator — generates synthetic ML training datasets from Unreal Engine 5.7+ scenes using Movie Render Queue (MRQ). Supports five output formats: DOPE (6DoF pose), YOLO detection, YOLO segmentation, YOLO OBB, and test images.

## Two Execution Environments

**Generator scripts** (`generate_*.py`, `yolo_v3/generate_yolo_v3.py`) run **inside the UE5 Editor** Python environment using the `unreal` module. They cannot be run standalone.

**Verification scripts** (`verify_*.py`), **merge_datasets.py**, and **modeltest/** run as **standalone Python** with `pillow`, `numpy`, `opencv-python`.

Install opencv in UE5's bundled Python:
```
<UE5_ROOT>/Engine/Binaries/ThirdParty/Python3/Win64/python.exe -m pip install opencv-python-headless
```

## Configuration System

- `config.py` — local config (gitignored), copy from `config_template.py`
- Settings are prefixed per generator: `DOPE_*`, `YOLO_*`, `SEG_*`, `OBB_*`, `TEST_*`, `YOLO_V3_*`
- Shared settings: `TARGET_TAG`, `CAMERA_TAG`, `IGNORE_TAG`, `POOL_BOUNDS`, camera intrinsics, MRQ params
- All generators hot-reload config via `importlib.reload(sys.modules['config'])`

## Architecture

### Generator Pipeline (all generators follow this pattern)
1. Find actors by tag (`TrainObject`, `AUV_Camera`, `IgnoreObject`)
2. Sample camera positions (hemisphere sampling within `POOL_BOUNDS`)
3. Generate annotations **before** rendering (pre-render label generation)
4. Create LevelSequence with camera keyframes
5. MRQ renders images; post-processing removes gap frames

### SceneCapture2D Two-Pass Differential Masking
Used by segmentation/detection scripts for occlusion-aware masks:
1. Hide target → capture background
2. Show target → capture foreground
3. `cv2.absdiff(fg, bg)` → visible-only mask
4. Morphological cleanup → `cv2.findContours()` → `cv2.approxPolyDP()`

### Anti-Ghosting (critical — do not change)
- TAA disabled, use FXAA: `aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_FXAA`
- `TEMPORAL_SAMPLES = 1` (never increase)
- All temporal effects disabled via console commands

### YOLO V3 Pipeline (`yolo_v3/`)
Registry-driven per-object dataset generation with automatic merging:
- `object_registry.py` — single source of truth for object definitions (actor labels, camera groups, hemisphere type, sample counts, co-visible objects)
- `generate_yolo_v3.py` — per-object generator with async MRQ callbacks
- `merge_datasets.py` — combines single-class datasets into multi-class with automatic class ID remapping (alphabetical ordering)
- Composite setups (e.g., slalom): use `skip_target_bbox` to orbit an invisible anchor without labeling it; use `class_name` to override data.yaml class names so composite and standalone datasets merge to the same class
- Sub-actors are fully keyframed in the LevelSequence (above ground for positive frames, underground for negatives) and excluded from `_hide_non_targets`

### Coordinate Systems
- **UE5**: X-forward, Y-right, Z-up (cm)
- **OpenCV/DOPE**: X-right, Y-down, Z-forward (meters) — see `ue_to_opencv_location()`, `ue_rotation_to_quaternion_xyzw()`
- **YOLO**: all coordinates normalized [0, 1] relative to image dimensions

## UE5 Scene Tags
- `TrainObject` — objects to detect/pose/segment
- `AUV_Camera` — CineCameraActor for data capture
- `IgnoreObject` — destroyed before rendering
- `HideInNegative` — actors that cannot use `TrainObject` (e.g., gate skeleton which uses colliders). Hidden during negative frames, and also hidden entirely when generating for unrelated objects. Use the `keep_visible` registry field to list which `HideInNegative` actor labels should stay visible for a given object.

## Running Verification Scripts

```bash
python verify_dope_data.py --data_path D:/UE5_Data/ --max_images 10
python verify_yolo_data.py --data_path D:/UE5_YOLO_Data/ --max_images 10
python verify_yolo_seg_data.py --data_path D:/UE5_YOLO_Seg_Data/ --max_images 10
python verify_obb_data.py --data_path D:/UE5_OBB_Data/ --max_images 10
```

## Merging Datasets (YOLO V3)

```bash
python yolo_v3/merge_datasets.py --source_root C:/data/ --groups cam_front --output C:/merged/
python yolo_v3/merge_datasets.py --source_root C:/data/ --all --output C:/merged_all/
```

## Key Constraints

- DOPE cuboid corner ordering follows NDDS standard: `dope_order = [5, 1, 2, 6, 4, 0, 3, 7]`
- Visibility filtering: DOPE requires >=20% visible pixels; YOLO filters below `MIN_VISIBLE_PIXELS`
- `OUTPUT_FOLDER` is **cleared** on each generator run
- Camera hemispheres: "horizontal" for front-facing (phi ~60-90°), "vertical" for bird's-eye (full upper hemisphere)
- Class IDs in merged datasets are assigned alphabetically

## UE5 Python API Patterns Used

### World & Subsystems
```python
# Preferred: UnrealEditorSubsystem (UE5.1+)
subsys = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = subsys.get_editor_world()
# Fallback (older API):
world = unreal.EditorLevelLibrary.get_editor_world()

# Actor subsystem (spawn, enumerate)
subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
actors = subsys.get_all_level_actors()
actor = subsys.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(), unreal.Rotator())
```

### Actor Queries & Tags
```python
actor.actor_has_tag("TagName")          # check tag
actor.get_actor_label()                 # display name (used as class name)
actor.get_actor_location()              # → unreal.Vector
actor.get_actor_rotation()              # → unreal.Rotator
actor.get_actor_bounds(False)           # → (origin: Vector, extent: Vector) half-extents
actor.get_components_by_class(unreal.BoxComponent)  # component query
comp.component_has_tag("TagName")
comp.get_world_location()
actor.set_actor_location_and_rotation(pos, rot, sweep=False, teleport=True)
actor.set_actor_hidden_in_game(True/False)  # used in two-pass masking
actor.destroy_actor()
```

### Math Library
```python
unreal.MathLibrary.find_look_at_rotation(from_vec, to_vec)  # → Rotator
unreal.MathLibrary.transform_location(transform, point)      # world↔local
transform.inverse()                     # inverse transform for camera-space projection
unreal.Transform(location=vec, rotation=rot)
unreal.Vector(x, y, z)
unreal.Rotator(roll, pitch, yaw)
unreal.FrameNumber(int)
unreal.FrameRate(24, 1)
```

### Asset Tools
```python
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
asset = asset_tools.create_asset(asset_name, pkg_path, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
unreal.EditorAssetLibrary.does_asset_exist("/Game/Path/Asset")
unreal.EditorAssetLibrary.delete_asset("/Game/Path/Asset")
unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
```

### LevelSequence / MRQ
```python
# Sequence setup
seq.set_display_rate(unreal.FrameRate(24, 1))
seq.set_playback_start(0); seq.set_playback_end(total_frames)
binding = seq.add_possessable(actor)
track = binding.add_track(unreal.MovieScene3DTransformTrack)
section = track.add_section()
section.set_range(start, end)
channels = section.get_all_channels()  # [tx, ty, tz, roll, pitch, yaw]
channels[i].add_key(unreal.FrameNumber(n), value)
key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_CONSTANT)

# Camera cut track
movie_scene = seq.get_movie_scene()
cut_track = movie_scene.add_camera_cut_track()  # preferred
# fallback: seq.add_track(unreal.MovieSceneCameraCutTrack)
cam_binding_id = unreal.MovieSceneObjectBindingID()
cam_binding_id.set_editor_property("guid", cam_binding.get_id())
cut_section.set_camera_binding_id(cam_binding_id)

# MRQ execution
mrq = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
queue = mrq.get_queue()
queue.delete_all_jobs()
job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
job.sequence = unreal.SoftObjectPath(seq.get_path_name())
job.map = unreal.SoftObjectPath(world.get_path_name())
config = job.get_configuration()
config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)

output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
output.output_directory = unreal.DirectoryPath(path)
output.output_resolution = unreal.IntPoint(W, H)
output.file_name_format = "{frame_number}"
output.zero_pad_frame_numbers = 6
output.flush_disk_writes_per_shot = True
output.use_custom_playback_range = True
output.custom_start_frame = 0; output.custom_end_frame = N
output.handle_frame_count = 0

aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
aa.spatial_sample_count = 1; aa.temporal_sample_count = 1
aa.override_anti_aliasing = True
aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_FXAA  # NEVER change — anti-ghosting
aa.render_warm_up_count = N; aa.engine_warm_up_count = N; aa.render_warm_up_frames = True

game = config.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
game.cinematic_quality_settings = True
game.texture_streaming = unreal.MoviePipelineTextureStreamingMethod.DISABLED
game.use_lod_zero = True; game.disable_hlods = True

console = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
console.start_console_commands = ["r.TemporalAA 0", ...]

executor = unreal.MoviePipelinePIEExecutor()
executor.on_executor_finished_delegate.add_callable(on_finished_callback)
mrq.render_queue_with_executor_instance(executor)
# IMPORTANT: keep `executor` in a global to prevent GC during render
```

### SceneCapture2D (Two-Pass Masking)
```python
cc = capture_actor.capture_component2d
cc.texture_target = render_target
cc.set_editor_property('primitive_render_mode', unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES)
cc.set_editor_property('capture_source', unreal.SceneCaptureSource.SCS_BASE_COLOR)
cc.set_editor_property('fov_angle', fov_degrees)
cc.set_editor_property('capture_every_frame', False)
cc.set_editor_property('capture_on_movement', False)
cc.set_world_location_and_rotation(pos, rot, False, True)
cc.capture_scene()  # manual trigger

# Read pixels
colors = unreal.RenderingLibrary.read_render_target(world, render_target, normalize=True)
# colors is a flat list of LinearColor; iterate with idx//w, idx%w for (y, x)
```

### CineCameraActor Configuration
```python
cine_comp = camera_actor.get_cine_camera_component()
cine_comp.filmback.sensor_width = SENSOR_WIDTH_MM
cine_comp.filmback.sensor_height = SENSOR_HEIGHT_MM
cine_comp.current_focal_length = FOCAL_LENGTH_MM
cine_comp.focus_settings.focus_method = unreal.CameraFocusMethod.DISABLE
fov_degrees = cine_comp.field_of_view  # read computed FOV
actual_pos = cine_comp.get_world_location()  # component may offset from actor root
```

### Render Target Creation
```python
# Strategy 1 (preferred, UE5.1+)
rt = unreal.RenderingLibrary.create_render_target2d(
    world, width=W, height=H,
    format=unreal.TextureRenderTargetFormat.RTF_RGBA8,
    clear_color=unreal.LinearColor(0, 0, 0, 1)
)
# Strategy 2 (fallback via factory)
factory = unreal.CanvasRenderTarget2DFactoryNew()
rt = asset_tools.create_asset(name, pkg, unreal.CanvasRenderTarget2D, factory)
rt.set_editor_property('size_x', W); rt.set_editor_property('size_y', H)
rt.set_editor_property('render_target_format', unreal.TextureRenderTargetFormat.RTF_RGBA8)
```

### Logging & UI
```python
unreal.log("message")           # info
unreal.log_warning("message")   # warning
unreal.log_error("message")     # error
with unreal.ScopedSlowTask(total, "Label...") as slow_task:
    slow_task.make_dialog(True)   # show cancel button
    slow_task.enter_progress_frame(1)
    if slow_task.should_cancel(): break
```
