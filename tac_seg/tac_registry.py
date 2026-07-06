# =============================================================================
# TAC Segmentation Registry — Single Source of Truth (TAC Challenge)
# =============================================================================
# Slim, fast sibling of yolo_v3/object_registry.py for the TAC Challenge
# underwater-ocean scene. Because nothing occludes the camera→target line in
# this environment, segmentation labels are produced by ANALYTIC PROJECTION
# (project the annotation volume, take the 2D convex hull) instead of the
# SceneCapture two-pass differential mask. No GPU captures, no pixel reads.
#
# No UE5 dependency — importable by both the generator (in-engine) and any
# standalone tooling.
#
# Valid TAC_SEG_GENERATE values (used in config.py):
#   Individual: "pipeline"
#   Groups:     "ocean"
#   Special:    "all"
# =============================================================================


# ---------------------------------------------------------------------------
# Default values — applied to any field not specified per-object
# ---------------------------------------------------------------------------
DEFAULTS = {
    "samples": 200,
    "min_distance": 100.0,
    "max_distance": 400.0,
    "hemisphere": "horizontal",     # "vertical" or "horizontal"
    "placement": None,              # None = no per-frame XY/rotation randomization
    "rotation_dr": None,            # None = no per-frame rotational sway
    "negative_ratio": 0.1,
    "val_split": 0.2,
    "enable_jitter": True,
    "jitter_max_pitch": 5.0,
    "jitter_max_yaw": 5.0,
    "samples_on_edges": True,       # False = hide co/sub actors whose bbox hits an edge
    "silhouette": "box",            # "box" (convex hull of OBB) or "cylinder"
}


# ---------------------------------------------------------------------------
# Per-object definitions
# ---------------------------------------------------------------------------
# Each key is the object's canonical name (folder name + YOLO class name).
#
#   "actor_label"        — World Outliner name. Defaults to the canonical key.
#   "camera_group"       — output sub-folder; default hemisphere hint.
#   "hemisphere"         — "horizontal" (orbit at eye level) or "vertical"
#                          (orbit overhead, bird's-eye).
#   "samples"            — number of positive frames.
#   "min_distance"/"max_distance" — camera distance range (cm).
#   "phi_max"            — vertical hemisphere: max polar angle from top (deg).
#   "theta_range"        — horizontal hemisphere: (min_deg, max_deg) azimuth.
#   "negative_ratio"     — fraction of empty/background frames.
#   "val_split"          — validation fraction.
#   "class_name"         — override the class name written to data.yaml
#                          (lets composite + standalone datasets merge).
#   "orbit_anchor_label" — orbit/aim a separate scene actor while labeling
#                          actor_label (use with skip_target_bbox for an
#                          invisible center anchor).
#   "skip_target_bbox"   — True = the target is an anchor, gets no label.
#   "co_visible"         — list of canonical names kept visible + labeled as
#                          their own class (multi-class in one render).
#   "sub_actors"         — list of actor labels labeled as the TARGET's class.
#   "samples_on_edges"   — False = hide co/sub actors touching an image edge.
#
#   "silhouette"         — "box" (default): label polygon = convex hull of the
#                          projected oriented bounding box. Good for boxy or
#                          convex shapes. "cylinder": derive a pipe-like axis +
#                          radius from the annotation box and project rim points
#                          for a tighter capsule silhouette (recommended for
#                          round pipes). Optional tuning via "silhouette_opts":
#                            {"segments": 16}   # angular samples around the axis
#
#   "placement"          — dict to randomize the object per positive frame:
#                            {"xy_range_x", "xy_range_y", "yaw_range",
#                             "roll_range", "pitch_min", "pitch_max"}
#   "rotation_dr"        — dict for rotational sway about the object's bottom:
#                            {"mode": "bottom_pivot", "roll_range", "pitch_range",
#                             "apply_to_self", "apply_to_sub_actors"}
# ---------------------------------------------------------------------------

OBJECT_DEFS = {
    # =========================================================================
    # ocean group — TAC Challenge underwater pipeline segment
    # =========================================================================
    "pipeline": {
        "actor_label": "pipeline",      # ← match the World Outliner name
        "camera_group": "ocean",
        "class_id": 0,
        "hemisphere": "horizontal",
        "samples": 1500,
        "min_distance": 150.0,
        "max_distance": 600.0,
        "silhouette": "cylinder",       # round pipe → capsule silhouette
        "silhouette_opts": {"segments": 16},
        # Gentle sway so the pipe is seen at varied roll/pitch without moving
        # its footing. Comment out for a perfectly static pipe.
        "rotation_dr": {
            "mode": "bottom_pivot",
            "roll_range": 6.0,
            "pitch_range": 6.0,
            "apply_to_self": True,
            "apply_to_sub_actors": True,
        },
        # Example multi-segment setup: uncomment if several pipe segments share
        # the scene and should all be labeled under the same class.
        # "sub_actors": ["pipeline_2", "pipeline_3"],
    },
}


# ---------------------------------------------------------------------------
# Camera groups — maps group name to list of object names
# ---------------------------------------------------------------------------
CAMERA_GROUPS = {
    "ocean": [
        "pipeline",
    ],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_object_config(name):
    """Return a merged config dict (defaults + per-object overrides).

    Raises:
        KeyError: if the object name is not in the registry.
    """
    if name not in OBJECT_DEFS:
        raise KeyError(
            f"Object '{name}' not found in registry. "
            f"Valid names: {sorted(OBJECT_DEFS.keys())}"
        )

    merged = dict(DEFAULTS)
    merged.update(OBJECT_DEFS[name])

    if merged.get("samples", 0) == 0:
        merged["samples"] = DEFAULTS["samples"]
    if "actor_label" not in merged:
        merged["actor_label"] = name
    return merged


def resolve_targets(selection_list):
    """Expand a TAC_SEG_GENERATE selection into individual object names.

    Supports:
        ["all"]              → every object in the registry
        ["ocean"]            → all objects in the ocean group
        ["pipeline"]         → just that object

    Raises:
        ValueError: if a name is not a valid object or group.
    """
    result = []
    for item in selection_list:
        if item == "all":
            result.extend(k for k, v in OBJECT_DEFS.items() if not v.get("helper"))
        elif item in CAMERA_GROUPS:
            result.extend(CAMERA_GROUPS[item])
        elif item in OBJECT_DEFS:
            result.append(item)
        else:
            raise ValueError(
                f"Unknown target '{item}'. "
                f"Valid objects: {sorted(OBJECT_DEFS.keys())}. "
                f"Valid groups: {sorted(CAMERA_GROUPS.keys())}. "
                f"Or use 'all'."
            )

    seen = set()
    ordered = []
    for name in result:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def get_all_class_names():
    """All canonical object names, sorted (defines merge class-ID order)."""
    return sorted(OBJECT_DEFS.keys())
