# UE5 Synthetic Data Generator

Generate synthetic training datasets from Unreal Engine 5.7+ scenes using Movie Render Queue (MRQ). Supports four output formats for different ML tasks.

## Supported Formats

| Script | ML Task | Label Format |
|--------|---------|-------------|
| `generate_dope_data.py` | 6DoF Pose Estimation | JSON — `{camera_data, objects[{location, quaternion_xyzw, projected_cuboid}]}` |
| `generate_yolo_data.py` | 2D Object Detection | TXT — `class_id x_center y_center width height` |
| `generate_yolo_seg_data.py` | Instance Segmentation | TXT — `class_id x1 y1 x2 y2 x3 y3 ...` |
| `generate_obb_data.py` | Oriented Bounding Box | TXT — `class_id x1 y1 x2 y2 x3 y3 x4 y4` |
| `generate_test_images.py` | Model Inference Testing | Images only (no annotations) |

## Prerequisites

### UE5 Scene Setup

1. Open your project in Unreal Engine 5.7+
2. Configure tags in the editor (Select actor → Details → Tags):
   - **`TrainObject`**: Add to any objects you want to detect, pose, or segment.
   - **`AUV_Camera`**: Add to the CineCamera actor used for capturing data.
   - **`IgnoreObject`**: Add to any background/debug objects you want the generator to completely ignore and destroy before rendering.
   - **`DOPE_Bounds`** *(component tag)*: Add to a `BoxComponent` on an actor to override its default bounding box for DOPE cuboid computation. If not present, the generator falls back to the static mesh bounds or `get_actor_bounds()`. Useful for actors with complex geometry where the automatic bounds don't represent the ideal cuboid for pose estimation.

   > **Note:** Some objects use invisible collision meshes with separate box colliders because they are part of a composite mesh that can't be split into individual actors. In these cases, `DOPE_Bounds` is required — without it, the generator would compute bounds from the entire composite mesh rather than the intended sub-object.

### Python (for verification scripts)

```bash
pip install pillow numpy
```

## Generating Datasets (inside UE5)

> ⚠️ Generator scripts must run **inside the UE5 Editor**. They use the `unreal` Python module and MRQ, which require the editor runtime.

### Steps

1. Open your project in UE5 Editor
2. Open the Python console: **Tools → Python Console** (or Output Log → Python tab)
3. Configure the script constants (edit the `.py` file before running):
   - `OUTPUT_FOLDER` — where images and labels are saved (directory is **cleared** on each run)
   - `NUM_SAMPLES` — number of images to generate
   - `MIN_DISTANCE` / `MAX_DISTANCE` — camera distance range from target (cm)
   - `POOL_BOUNDS` — 3D volume constraints for camera positions
   - `FOCAL_LENGTH_MM`, `SENSOR_WIDTH_MM`, `SENSOR_HEIGHT_MM` — camera intrinsics
4. Run the script:

```python
# DOPE (6DoF Pose Estimation)
exec(open("C:/code/UE5_SyntheticDataGen/generate_dope_data.py").read())

# YOLO Detection
exec(open("C:/code/UE5_SyntheticDataGen/generate_yolo_data.py").read())

# YOLO Segmentation
exec(open("C:/code/UE5_SyntheticDataGen/generate_yolo_seg_data.py").read())

# YOLO OBB
exec(open("C:/code/UE5_SyntheticDataGen/generate_obb_data.py").read())

# Test Images (no annotations, for model inference)
exec(open("C:/code/UE5_SyntheticDataGen/generate_test_images.py").read())
```

5. Wait for **"RENDER COMPLETE!"** in the Output Log before running another script

### Output Structure

**DOPE:**
```
D:/UE5_Data/
├── 000000.png
├── 000000.json
├── 000001.png
├── 000001.json
└── ...
```

**YOLO / YOLO Seg / OBB:**
```
D:/UE5_YOLO_Data/          (or UE5_YOLO_Seg_Data, UE5_OBB_Data)
├── images/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── labels/
│   ├── 000000.txt
│   ├── 000001.txt
│   └── ...
└── classes.txt
```

**Test Images:**
```
D:/UE5_Test_Images/
├── 000000.png
├── 000001.png
└── ...
```

## Verifying Datasets (standalone Python)

> Verification scripts run **outside UE5** with standard Python. They overlay annotations on rendered images so you can visually confirm correctness.

```bash
# DOPE — overlays 3D cuboid projections
python verify_dope_data.py --data_path D:/UE5_Data/ --max_images 10

# YOLO Detection — overlays 2D bounding boxes
python verify_yolo_data.py --data_path D:/UE5_YOLO_Data/ --max_images 10

# YOLO Segmentation — overlays polygon masks
python verify_yolo_seg_data.py --data_path D:/UE5_YOLO_Seg_Data/ --max_images 10

# YOLO OBB — overlays oriented bounding boxes
python verify_obb_data.py --data_path D:/UE5_OBB_Data/ --max_images 10
```

Overlay images are saved to `{data_path}/verify/`. Open them and confirm:
- **DOPE**: Colored dots align with object corners (9 points per object)
- **YOLO Detect**: Rectangles tightly bound the objects
- **YOLO Seg**: Polygons tightly follow the object silhouette (SceneCapture2D per-actor mask rendering)
- **OBB**: Rotated rectangles fit the objects tightly

## Validating with Ultralytics (YOLO formats)

After generation and visual verification, validate that Ultralytics can parse your dataset:

```bash
yolo detect val data=your_dataset.yaml   # Detection
yolo segment val data=your_dataset.yaml  # Segmentation
yolo obb val data=your_dataset.yaml      # OBB
```

## Format References

- [NVIDIA DOPE](https://github.com/NVlabs/Deep_Object_Pose) — Deep Object Pose Estimation
- [Ultralytics Detection Format](https://docs.ultralytics.com/datasets/detect/)
- [Ultralytics Segmentation Format](https://docs.ultralytics.com/datasets/segment/)
- [Ultralytics OBB Format](https://docs.ultralytics.com/datasets/obb/)

TODO: Fix obb- dope- yolo- for masks still being shown even when objects are not visible/half visible
TODO: Transition to double half sphered camera model for reliable 360 degree coverage
TODO: Add different variables for more accurate and realistic data generation