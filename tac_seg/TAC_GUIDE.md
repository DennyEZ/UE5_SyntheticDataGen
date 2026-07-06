# TAC Seg — Fast Analytic Segmentation Generator (TAC Challenge)

A lean, fast sibling of `yolo_v3/` for the **TAC Challenge underwater-ocean**
scene. It generates YOLO **segmentation** datasets, one class per object folder,
and is built on one simplifying assumption:

> **Nothing ever occludes the line between the camera and the target.**

That lets us delete the slowest part of the `yolo_v3` pipeline — the
SceneCapture2D **two-pass differential mask** (hide → capture bg → show →
capture fg → `cv2.absdiff` → contour) — and replace it with **analytic hull
projection**:

```
project the object's annotation volume → 2D
take the convex hull → clip to the image rectangle
→ YOLO segment polygon
```

**No GPU captures. No per-pixel render-target reads. No cv2 needed for labels.**
(`cv2` is only used for the optional downsample + JPG post-processing.)

---

## Why it's faster than `yolo_v3` segment mode

| Stage | `yolo_v3` segment | `tac_seg` |
|---|---|---|
| Mask per labeled actor | 2× `SceneCapture.capture_scene()` + 2× full-res Python pixel-read loop | pure CPU math (project ~32 points, convex hull) |
| GPU work for labels | yes (one render target per pass) | none |
| `cv2` for labels | required | not used |
| Occlusion handling | differential subtraction | none (assumed clear view) |

For N labeled actors per frame, `yolo_v3` does `2N` GPU captures **plus** `2N`
O(W·H) Python pixel loops. `tac_seg` does `0`. The label pass goes from the
dominant cost to negligible; render time (MRQ) becomes the only real cost.

---

## 1. New level — yes, make one

The generator is **level-agnostic**: it finds the camera + targets by tag in
whatever map is open and renders `world.get_path_name()`. So the ocean level is
purely a content asset — the Python doesn't reference it.

Create a separate ocean `.umap` because everything that differs lives in
**config + the level**, not the script:

- `TAC_OCEAN_BOUNDS` — camera-sampling volume (open water is larger than the pool)
- ocean lighting / fog / water material / floor / skybox

Keep it isolated from the RoboSub pool so neither competition's settings bleed
into the other.

### Scene requirements (same tag scheme as `yolo_v3`)

| What | Where | Example |
|---|---|---|
| Target tag | Details → Tags → `TrainObject` | the pipeline segment |
| Camera tag | Details → Tags → `AUV_Camera` | your CineCameraActor |
| Ignore tag | Details → Tags → `IgnoreObject` | actors destroyed before render |
| Actor label | World Outliner name | must match `actor_label` in the registry |

> Optional but recommended for a tight pipe mask: add a `BoxComponent` tagged
> `DOPE_Bounds` sized to the pipe. The generator uses it as the annotation
> volume; otherwise it falls back to the StaticMesh bounds, then the actor AABB.

---

## 2. Configure — `config.py` (copy from `config_template.py`)

```python
TAC_SEG_GENERATE   = ["pipeline"]          # or ["ocean"] / ["all"]
TAC_SEG_OUTPUT_ROOT = "C:/UE5_TAC_Seg_Data/"
TAC_OCEAN_BOUNDS = {                         # match your ocean .umap volume
    "x_min": -2000.0, "x_max": 2000.0,
    "y_min": -2000.0, "y_max": 2000.0,
    "z_min": -2000.0, "z_max": -1000.0,
}
TAC_SEG_DOWNSAMPLE_TO = (960, 540)           # render high, area-average down
TAC_SEG_IMAGE_FORMAT  = "jpg"
```

Shared camera intrinsics (`SENSOR_*`, `FOCAL_LENGTH_MM`, `RESOLUTION_*`) and
`WARMUP_FRAMES`/`SPATIAL_SAMPLES`/`TEMPORAL_SAMPLES` are reused from the top of
`config.py`. Anti-ghosting MRQ settings are identical to `yolo_v3` (FXAA,
`TEMPORAL_SAMPLES = 1`, temporal effects off) — **do not change them**.

---

## 3. Define objects — `tac_seg/tac_registry.py`

```python
"pipeline": {
    "actor_label": "pipeline",       # World Outliner name
    "camera_group": "ocean",         # output sub-folder
    "hemisphere": "horizontal",      # "horizontal" (eye-level orbit) | "vertical" (top-down)
    "samples": 1500,
    "min_distance": 150.0,
    "max_distance": 600.0,
    "silhouette": "cylinder",        # tighter capsule mask for round pipes
    "silhouette_opts": {"segments": 16},
    "rotation_dr": {                 # optional gentle sway
        "mode": "bottom_pivot",
        "roll_range": 6.0, "pitch_range": 6.0,
        "apply_to_self": True, "apply_to_sub_actors": True,
    },
},
```

### Silhouette modes
- `"box"` (default) — label polygon = convex hull of the projected oriented
  bounding box. Best for boxy/convex objects.
- `"cylinder"` — derives a pipe axis + radius from the annotation box and
  projects rim points at both ends, giving a capsule silhouette that hugs a
  round pipe far better than 8 box corners. The longest box dimension is the
  axis; the radius circumscribes the cross-section (convex over-cover of the
  rounded caps). Tune density with `silhouette_opts={"segments": N}`.

> Both modes produce a **convex** polygon. For genuinely concave shapes,
> analytic projection over-covers — use `yolo_v3` segment mode (mask-based)
> instead.

### Supported registry fields
`actor_label`, `camera_group`, `hemisphere`, `samples`, `min_distance`,
`max_distance`, `phi_max` (vertical), `theta_range` (horizontal),
`negative_ratio`, `val_split`, `enable_jitter`, `jitter_max_pitch`,
`samples_on_edges`, `silhouette`/`silhouette_opts`, `class_name`,
`orbit_anchor_label`, `skip_target_bbox`, `co_visible` (own class),
`sub_actors` (target's class), `placement`, `rotation_dr`.

> Deliberately **not** ported from `yolo_v3` (RoboSub-pool-specific): two-pass
> masking, occlusion modes, `HideInNegative`/`keep_visible`,
> `keep_visible_unlabeled`, `hard_negative_actors`, `variant_tags`, slot-group
> placement, `mask_*` overrides.

---

## 4. Run (inside the UE5 Editor)

1. Open the **ocean** level.
2. Verify the pipeline has `TrainObject` + correct label, and the camera has `AUV_Camera`.
3. Set `config.py` as above.
4. Window → Output Log, then run:
   ```
   py "C:/.../UE5_SyntheticDataGen/tac_seg/generate_tac_seg.py"
   ```
5. Watch the progress bar (label pass is near-instant) and the MRQ render.

### Output
```
C:/UE5_TAC_Seg_Data/
└── ocean/
    └── pipeline/
        ├── data.yaml          (task: segment, nc: 1, names: {0: pipeline})
        ├── train/images & labels/
        └── val/images & labels/
```

### Crash recovery
If interrupted mid-run, actors may be left underground. Restore them:
```python
from tac_seg.generate_tac_seg import restore_scene
restore_scene()
```

---

## 5. Merge / verify (standalone Python — reuse `yolo_v3` tools)

The output is standard YOLO-seg, so the existing tools work unchanged:

```bash
# verify
python verify_yolo_seg_data.py --data_path C:/UE5_TAC_Seg_Data/ocean/pipeline/ --max_images 10

# merge multiple TAC classes into one multi-class set
python yolo_v3/merge_datasets.py --source_root C:/UE5_TAC_Seg_Data/ --groups ocean --output C:/tac_merged/
```

---

## 6. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `No camera with tag 'AUV_Camera'` | camera not tagged | add `AUV_Camera` to the CineCameraActor |
| `Actor 'pipeline' not found` | label mismatch | match World Outliner name to `actor_label` |
| Mask is a loose box, not a pipe | `silhouette` is `"box"` | set `"silhouette": "cylinder"` |
| Mask edges look faceted | too few rim samples | raise `silhouette_opts={"segments": 24}` |
| Mask slightly larger than the pipe | cylinder radius over-covers caps | expected (convex over-cover); reduce the `DOPE_Bounds` box cross-section |
| Camera samples wrong area in negatives | wrong bounds | set `TAC_OCEAN_BOUNDS` to the ocean map's volume |
| JPG/downsample skipped | OpenCV missing in UE5 Python | `pip install opencv-python-headless` into UE5's Python |
```
