# UE5 Synthetic Data Generator - Copilot Instructions

## Project Overview
This project generates synthetic training datasets using Unreal Engine 5.7+ and Movie Render Queue (MRQ). It supports multiple output formats for different ML frameworks.

## Generator Scripts

| Script | Output Format | Use Case | Documentation |
|--------|--------------|----------|---------------|
| `generate_dope_data.py` | JSON (DOPE format) | 6DoF pose estimation | [NVIDIA DOPE](https://github.com/NVlabs/Deep_Object_Pose) |
| `generate_yolo_data.py` | TXT (YOLO detection) | 2D object detection | [Ultralytics Detect](https://docs.ultralytics.com/datasets/detect/) |
| `generate_yolo_seg_data.py` | TXT (YOLO segmentation) | Instance segmentation | [Ultralytics Segment](https://docs.ultralytics.com/datasets/segment/) |
| `generate_obb_data.py` | TXT (YOLO OBB) | Oriented bounding boxes | [Ultralytics OBB](https://docs.ultralytics.com/datasets/obb/) |

## Output Formats

### DOPE (generate_dope_data.py)
```
OUTPUT_FOLDER/
├── 000000.png
├── 000000.json  # {camera_data, objects[{class, location[3], quaternion_xyzw[4], projected_cuboid[9][2]}]}
```

### YOLO Detection (generate_yolo_data.py)
```
OUTPUT_FOLDER/
├── images/000000.png
├── labels/000000.txt  # class_id x_center y_center width height (normalized 0-1)
├── classes.txt
```

### YOLO Segmentation (generate_yolo_seg_data.py)
```
OUTPUT_FOLDER/
├── images/000000.png
├── labels/000000.txt  # class_id x1 y1 x2 y2 x3 y3 ... (polygon vertices, normalized)
├── classes.txt
```

### YOLO OBB (generate_obb_data.py)
```
OUTPUT_FOLDER/
├── images/000000.png
├── labels/000000.txt  # class_id x1 y1 x2 y2 x3 y3 x4 y4 (4 corners, normalized)
├── classes.txt
```

## Architecture

### All Generator Scripts (UE5 Editor Python)
- Run inside Unreal Engine's Python environment (uses `unreal` module)
- Find actors by tag: `TrainObject` (objects), `AUV_Camera` (camera)
- Create LevelSequence with camera keyframes
- Generate annotations before rendering
- MRQ renders images; post-processing removes gap frames

### Verification Scripts (Standalone Python)
- `verify_dope_data.py` - Overlays 3D cuboid projections
- `verify_yolo_data.py` - Overlays 2D bounding boxes
- `verify_yolo_seg_data.py` - Overlays segmentation polygons
- `verify_obb_data.py` - Overlays oriented bounding boxes

## Critical Patterns

### Coordinate System (DOPE only)
```
UE5: X-forward, Y-right, Z-up (cm)
OpenCV: X-right, Y-down, Z-forward (meters)
```
See `ue_to_opencv_location()` and `ue_rotation_to_quaternion_xyzw()`.

### YOLO Normalization (YOLO variants)
All coordinates normalized to [0, 1] relative to image dimensions.

### Anti-Ghosting Configuration (ALL scripts)
Temporal artifacts are critical. These must remain disabled:
- TAA → use FXAA: `aa.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_FXAA`
- `TEMPORAL_SAMPLES = 1` (never increase)
- Console commands disable all temporal effects

### DOPE Cuboid Ordering
Corner ordering: `dope_order = [5, 1, 2, 6, 4, 0, 3, 7]` (DOPE/NDDS standard).

## Usage

### Running Generators (inside UE5)
1. Tag objects with `TrainObject` and camera with `AUV_Camera`
2. Configure `OUTPUT_FOLDER`, `NUM_SAMPLES`, camera parameters
3. Execute via UE5 Python console

### Verifying Output (standalone)
```bash
# DOPE
python verify_dope_data.py --data_path D:/UE5_Data/ --max_images 10

# YOLO Detection
python verify_yolo_data.py --data_path D:/UE5_YOLO_Data/ --max_images 10

# YOLO Segmentation
python verify_yolo_seg_data.py --data_path D:/UE5_YOLO_Seg_Data/ --max_images 10

# YOLO OBB
python verify_obb_data.py --data_path D:/UE5_OBB_Data/ --max_images 10
```

### Validating with Ultralytics (YOLO formats)
```bash
yolo detect val data=your_dataset.yaml   # Detection
yolo segment val data=your_dataset.yaml  # Segmentation
yolo obb val data=your_dataset.yaml      # OBB
```

## Configuration Reference

| Constant | Purpose |
|----------|---------|
| `OUTPUT_FOLDER` | Render output directory (cleared on run) |
| `WARMUP_FRAMES` | Frames to stabilize (64 recommended) |
| `MIN/MAX_DISTANCE` | Camera distance range (cm) |
| `POOL_BOUNDS` | 3D volume constraints for camera |
| `FOCAL_LENGTH_MM`, `SENSOR_*_MM` | Camera intrinsics |
