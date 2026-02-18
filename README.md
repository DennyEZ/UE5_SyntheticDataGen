# UE5 Synthetic Data Generator

Generate synthetic training datasets from Unreal Engine 5.7+ scenes using Movie Render Queue (MRQ). Supports four output formats for different ML tasks.

## Supported Formats

| Script | ML Task | Label Format |
|--------|---------|-------------|
| `generate_dope_data.py` | 6DoF Pose Estimation | JSON — `{camera_data, objects[{location, quaternion_xyzw, projected_cuboid}]}` |
| `generate_yolo_data.py` | 2D Object Detection | TXT — `class_id x_center y_center width height` |
| `generate_yolo_seg_data.py` | Instance Segmentation | TXT — `class_id x1 y1 x2 y2 x3 y3 ...` |
| `generate_obb_data.py` | Oriented Bounding Box | TXT — `class_id x1 y1 x2 y2 x3 y3 x4 y4` |

## Prerequisites

### UE5 Scene Setup

1. Open your project in Unreal Engine 5.7+
2. Tag target objects: Select actor → Details → Tags → add **`TrainObject`**
3. Tag camera: Select CineCamera → Details → Tags → add **`AUV_Camera`**

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
