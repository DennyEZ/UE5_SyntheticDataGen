# YOLO V3 Dataset Generation Pipeline — Complete Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    config.py (gitignored)                │
│  YOLO_V3_GENERATE, YOLO_V3_OUTPUT_ROOT, YOLO_V3_MODE   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│               object_registry.py                         │
│  Per-object: hemisphere, camera_group, samples,          │
│  distances, placement config                             │
│  Helpers: resolve_targets(), get_object_config()         │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌──────────────────────┐   ┌─────────────────────────────┐
│  generate_yolo_v3.py │   │     merge_datasets.py       │
│  (runs in UE5 editor)│   │  (runs in standard Python)  │
│                      │   │                             │
│  Per-object loop:    │   │  Combines single-class      │
│  1. Find actor       │   │  datasets into multi-class  │
│  2. Hide others      │   │  with ID remapping          │
│  3. Camera hemisphere│   │                             │
│  4. Generate labels  │   │  python merge_datasets.py   │
│  5. MRQ render       │   │    --source_root ... --all  │
│  6. Split train/val  │   │    --output ...             │
│  7. Next object      │   │                             │
└──────────┬───────────┘   └──────────────┬──────────────┘
           │                              │
           ▼                              ▼
   cam_front/                      merged_dataset/
   ├── gate_sawfish/               ├── data.yaml (nc=N)
   │   ├── data.yaml (nc=1)       ├── train/images & labels
   │   ├── train/                 └── val/images & labels
   │   └── val/
   ├── octagon/
   └── ...
```

---

## 1. Prerequisites

### UE5 Python Packages
```
<UE5_ROOT>/Engine/Binaries/ThirdParty/Python3/Win64/python.exe -m pip install opencv-python-headless
```
> Required for mask-based bounding boxes. Without it, the script falls back to fast AABB projection (less accurate).

### UE5 Scene Requirements
Every actor you want to generate data for needs **two things**:

| What | Where | Example |
|---|---|---|
| **Actor Tag** | Details panel → Tags → add `TrainObject` | All target objects get this tag |
| **Actor Label** | The name in the World Outliner | Must match `actor_label` in registry |
| **Camera Tag** | Details panel → Tags → add `AUV_Camera` | Your CineCameraActor |

---

## 2. Object Registry — `object_registry.py`

This is the **single source of truth** for all objects. Edit this file to:

### Add a new object
```python
OBJECT_DEFS = {
    # ...existing objects...

    "my_new_object": {                          # ← canonical name (used in folders/class names)
        "actor_label": "MyNewObject_BP",        # ← MUST match World Outliner name
        "camera_group": "cam_front",            # ← determines output folder and default hemisphere
        "hemisphere": "horizontal",             # ← "horizontal" (front cam) or "vertical" (bottom cam)
        "samples": 200,                         # ← number of positive samples
        "min_distance": 100.0,                  # ← camera distance range (cm)
        "max_distance": 400.0,
    },
}
```

Then add it to the appropriate camera group:
```python
CAMERA_GROUPS = {
    "cam_front": [..., "my_new_object"],       # ← add here
}
```

### Change sample count
```python
# For ALL objects (default):
DEFAULTS = {"samples": 200, ...}

# For ONE specific object:
"octagon": {"samples": 100, ...}
```

### Enable position randomization (bottle/ladle only)
```python
"bottle": {
    "placement": {
        "xy_range_x": 18.0,    # cm offset from original position
        "xy_range_y": 15.0,
        "yaw_range": 360.0,    # degrees
        "roll_range": 90.0,
        "pitch_min": -90.0,
        "pitch_max": 0.0,
    },
}
```
Objects without `"placement"` keep their original scene position — only camera angles vary.

### Enable co-visible objects (multi-class frames)
Objects that appear together in real life should be configured as co-visible so the model learns to detect them simultaneously:

```python
"gate_sawfish": {
    "co_visible": ["gate_shark"],    # gate_shark stays visible and gets labeled
    ...
},
"gate_shark": {
    "co_visible": ["gate_sawfish"],  # symmetric — both reference each other
    ...
},
```

**What happens with co_visible:**
- When generating `gate_sawfish`, `gate_shark` stays visible (not moved underground)
- Both objects get labeled in each frame: `gate_sawfish` = class 0, `gate_shark` = class 1
- The output `data.yaml` has `nc: 2` with both class names
- On negative frames, both objects go underground (empty label)
- The merger handles multi-class source datasets correctly — all class IDs get remapped

**Without co_visible (default):** each object is generated in complete isolation. Only the single target is visible. This is fine for objects that don't naturally co-occur.

**When to use co_visible:**
| Objects | co_visible? | Reason |
|---|---|---|
| gate_sawfish ↔ gate_shark | ✅ Yes | Same gate structure |
| bin_shark ↔ bin_sawfish ↔ octagon_table | Consider | Same bin area |
| bottle (alone) | ❌ No | Gripper task, usually alone |

### Composite setups with `skip_target_bbox` and `class_name`

For setups where multiple objects appear together (e.g., a slalom course), use an invisible anchor actor as the orbit center and put all real objects as `sub_actors` / `co_visible`:

```python
"slalom": {
    "actor_label": "slalom_center",       # invisible anchor at center of slalom
    "class_name": "red_pipe",             # overrides class name in data.yaml
    "camera_group": "cam_front",
    "hemisphere": "horizontal",
    "samples": 200,
    "min_distance": 500.0,                # far enough to see all pipes
    "max_distance": 1200.0,
    "skip_target_bbox": True,             # anchor gets no bounding box
    "co_visible": ["slalom_white_pipe"],  # white pipes = different class
    "sub_actors": ["red_pipe", "red_pipe_2", "red_pipe_3"],  # same class as target
},
"slalom_white_pipe": {
    "actor_label": "white_pipe",
    "class_name": "white_pipe",           # matches standalone white_pipe for merging
    "camera_group": "cam_front",
    "samples": 0,                         # never generated standalone
    "sub_actors": ["white_pipe_2", "white_pipe_3", "white_pipe_4",
                   "white_pipe_5", "white_pipe_6"],
},
```

**`skip_target_bbox`**: When `True`, the target actor (orbit center) is not labeled — only `sub_actors` and `co_visible` objects get bounding boxes. Use this when the orbit center is an invisible anchor.

**`class_name`**: Overrides the class name written to `data.yaml`. Without it, the registry key name is used (e.g., `slalom`). With `class_name: "red_pipe"`, the merger treats slalom data and standalone `red_pipe` data as the same class.

**UE5 scene requirements for slalom:**
| Actor Label | Tag | Purpose |
|---|---|---|
| `slalom_center` | `TrainObject` | Invisible anchor at geometric center |
| `red_pipe` | `TrainObject` | Red pipe 1 |
| `red_pipe_2` | `TrainObject` | Red pipe 2 |
| `red_pipe_3` | `TrainObject` | Red pipe 3 |
| `white_pipe` | `TrainObject` | White pipe 1 |
| `white_pipe_2`–`white_pipe_6` | `TrainObject` | White pipes 2–6 |

### Restrict hemisphere bounds
Both hemisphere types support optional bounds to prevent the camera from reaching undesirable angles.

**Horizontal hemisphere — `theta_range`** restricts the azimuthal (side-to-side) angle so the camera only approaches from certain directions. Angles are in degrees (0–360):
```python
"gate_sawfish": {
    "hemisphere": "horizontal",
    "theta_range": (105.0, 255.0),  # only approach from the front arc, avoid side-on views
    ...
},
```

**Vertical hemisphere — `phi_max`** caps the polar angle from vertical so the camera can't drop to the equator (which would produce front-on images instead of top-down). Default is 90° (full hemisphere). Lower values keep the camera more overhead:
```python
"octagon_table": {
    "hemisphere": "vertical",
    "phi_max": 70.0,  # camera stays at least 20° above horizon
    ...
},
```

| Parameter | Hemisphere | What it limits | Default |
|---|---|---|---|
| `theta_range` | horizontal | Azimuthal angle range (degrees) | Full 360° |
| `phi_max` | vertical | Max polar angle from top (degrees) | 90° (full hemisphere) |

### Valid values for `actor_label` matching
If your object's World Outliner name **doesn't match** the dict key, add an explicit `actor_label`:
```python
"gate_sawfish": {
    "actor_label": "GateSawfish_Blueprint",    # ← exact Outliner name
    ...
}
```
If omitted, `actor_label` defaults to the dict key (e.g., `"gate_sawfish"`).

---

## 3. Config — `config.py`

### Select what to generate
```python
# Generate everything:
YOLO_V3_GENERATE = ["all"]

# Generate one camera group:
YOLO_V3_GENERATE = ["cam_front"]

# Generate specific objects:
YOLO_V3_GENERATE = ["octagon", "gate_sawfish"]

# Mix groups and individuals:
YOLO_V3_GENERATE = ["cam_bottom", "bottle"]
```

### Set output location
```python
YOLO_V3_OUTPUT_ROOT = "C:/UE5_YOLO_Data_V3/"
```

### Choose detection or segmentation
```python
YOLO_V3_MODE = "detect"     # bounding boxes: class x_center y_center width height
YOLO_V3_MODE = "segment"    # polygons: class x1 y1 x2 y2 ... xn yn
```

### Other settings (shared with V2)
```python
SENSOR_WIDTH_MM = 36.0       # camera sensor — match your CineCamera
SENSOR_HEIGHT_MM = 20.25
FOCAL_LENGTH_MM = 30.0
RESOLUTION_X = 1920          # render resolution
RESOLUTION_Y = 1080
WARMUP_FRAMES = 64           # MRQ warm-up frames
```

---

## 4. Running the Generator (in UE5 Editor)

### Step-by-Step
1. Open your UE5 project with the scene loaded
2. Verify actors have `TrainObject` tags and correct labels
3. Edit `config.py`:
   ```python
   YOLO_V3_GENERATE = ["octagon"]   # ← what you want
   YOLO_V3_OUTPUT_ROOT = "C:/UE5_YOLO_Data_V3/"
   YOLO_V3_MODE = "detect"
   ```
4. In UE5, open the Output Log (Window → Output Log)
5. Run the script:
   ```
   py "C:/path/to/UE5_SyntheticDataGen/generate_yolo_v3.py"
   ```
6. Wait for the progress bar and MRQ render to complete
7. Check the Output Log for status messages

### What happens internally
For each object in `YOLO_V3_GENERATE`:
1. **Find** the actor in the scene by matching `actor_label`
2. **Hide** all other `TrainObject` actors (moved underground)
3. **Create** a LevelSequence with camera keyframes (hemisphere trajectory)
4. **Generate labels** using SceneCapture2D two-pass differential masks
5. **Render** via MRQ (async — callback starts next object when done)
6. **Post-process**: remove gap frames, split into train/val, write `data.yaml`
7. **Restore** hidden actors, proceed to next object

### Output structure
```
C:/UE5_YOLO_Data_V3/
├── cam_front/
│   ├── gate_sawfish/
│   │   ├── data.yaml       ← nc: 1, names: {0: gate_sawfish}
│   │   ├── train/
│   │   │   ├── images/     ← rendered PNG frames
│   │   │   └── labels/     ← YOLO format .txt files
│   │   └── val/
│   │       ├── images/
│   │       └── labels/
│   └── octagon/
│       └── ...
└── cam_bottom/
    └── ...
```

---

## 5. Running the Merger (standard Python terminal)

The merger is a standalone script — no UE5 needed. Run it from any Python environment.

### Merge a camera group
```bash
python merge_datasets.py \
    --source_root C:/UE5_YOLO_Data_V3/ \
    --groups cam_front \
    --output C:/merged_cam_front/
```

### Merge specific objects
```bash
python merge_datasets.py \
    --sources C:/UE5_YOLO_Data_V3/cam_front/octagon C:/UE5_YOLO_Data_V3/cam_bottom/bin_shark \
    --output C:/merged_custom/
```

### Merge everything
```bash
python merge_datasets.py \
    --source_root C:/UE5_YOLO_Data_V3/ \
    --all \
    --output C:/merged_all/
```

### Preview before merging (dry-run)
```bash
python merge_datasets.py \
    --source_root C:/UE5_YOLO_Data_V3/ \
    --all \
    --output C:/merged_all/ \
    --dry_run
```

### What the merger does
1. Reads `data.yaml` from each source folder → collects class names
2. Sorts all class names alphabetically → assigns sequential global IDs (0, 1, 2, ...)
3. Copies images with prefix: `{object_name}_{original_name}.png`
4. Rewrites label files with remapped class IDs
5. Generates unified `data.yaml` with all classes

### Merged output
```
C:/merged_cam_front/
├── data.yaml            ← nc: 7, all cam_front classes
├── train/
│   ├── images/          ← gate_sawfish_000001.png, octagon_000001.png, ...
│   └── labels/          ← remapped class IDs
└── val/
    ├── images/
    └── labels/
```

---

## 6. Verification

### Using existing verify script
```bash
python verify_yolo_data.py --data_path C:/UE5_YOLO_Data_V3/cam_front/octagon/ --split train --max_images 5
```

### For merged datasets
```bash
python verify_yolo_data.py --data_path C:/merged_cam_front/ --split train --max_images 10
```

---

## 7. Common Workflows

### Workflow A: Quick single-object test
```
1. Edit config.py:     YOLO_V3_GENERATE = ["octagon"]
2. Run in UE5:         py "path/to/generate_yolo_v3.py"
3. Verify:             python verify_yolo_data.py --data_path C:/UE5_YOLO_Data_V3/cam_front/octagon/
```

### Workflow B: Full front-cam dataset
```
1. Edit config.py:     YOLO_V3_GENERATE = ["cam_front"]
2. Run in UE5:         py "path/to/generate_yolo_v3.py"
3. Merge:              python merge_datasets.py --source_root C:/UE5_YOLO_Data_V3/ --groups cam_front --output C:/cam_front_merged/
4. Train:              yolo detect train data=C:/cam_front_merged/data.yaml model=yolo11n.pt
```

### Workflow C: Regenerate specific objects
```
1. Edit config.py:     YOLO_V3_GENERATE = ["gate_sawfish", "octagon"]
2. Run in UE5:         (only these 2 folders are overwritten, others untouched)
3. Re-merge:           python merge_datasets.py --source_root ... --groups cam_front --output ...
```

### Workflow D: Both camera models
```
1. Edit config.py:     YOLO_V3_GENERATE = ["all"]
2. Run in UE5:         (generates all objects across all camera groups)
3. Merge front:        python merge_datasets.py --source_root ... --groups cam_front --output C:/front_model/
4. Merge bottom:       python merge_datasets.py --source_root ... --groups cam_bottom --output C:/bottom_model/
5. Train both:         yolo detect train data=C:/front_model/data.yaml ...
                       yolo detect train data=C:/bottom_model/data.yaml ...
```

---

## 8. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Actor 'X' not found — skipping` | Actor label mismatch | Check World Outliner name matches `actor_label` in registry |
| `No camera with tag 'AUV_Camera' found` | Camera not tagged | Add `AUV_Camera` tag to your CineCameraActor |
| `Invalid GENERATE config` | Typo in object/group name | Check valid names listed at top of `object_registry.py` |
| Empty rendered images | Non-target actors not hidden properly | Verify only the target object has `TrainObject` tag visible |
| `cv2 not found` | OpenCV not installed in UE5 Python | Run pip install command from Prerequisites section |
| Merger finds 0 sources | Wrong `--source_root` path | Ensure path has `cam_front/object_name/data.yaml` structure |

---

## 9. File Reference

| File | Runs In | Purpose |
|---|---|---|
| `object_registry.py` | Both | Object definitions, camera groups, helpers |
| `config.py` | Both | User-local settings (gitignored) |
| `config_template.py` | — | Reference for config.py defaults |
| `generate_yolo_v3.py` | UE5 Editor | Per-object dataset generator |
| `merge_datasets.py` | Terminal | Dataset merger with class-ID remapping |
| `verify_yolo_data.py` | Terminal | Visual verification of detection datasets |
| `verify_yolo_seg_data.py` | Terminal | Visual verification of segmentation datasets |
