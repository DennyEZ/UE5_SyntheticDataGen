# =============================================================================
# Object Registry — Single Source of Truth for YOLO Dataset Generation
# =============================================================================
# Defines every object that can be generated, its camera group, hemisphere
# type, placement strategy, and per-object generation parameters.
#
# No UE5 dependency — importable by both the generator (in-engine) and
# the merger (standalone Python).
#
# Valid GENERATE values (used in config.py):
#   Individual: "gate_sawfish", "gate_shark", "red_pipe", "white_pipe",
#               "torpedo_map", "bin_whole", "octagon",
#               "bin_shark", "bin_sawfish", "octagon_table",
#               "bottle"
#   Groups:     "cam_front", "cam_bottom", "cam_bottom_seg"
#   Special:    "all"
# =============================================================================


# ---------------------------------------------------------------------------
# Default values — applied to any field not specified per-object
# ---------------------------------------------------------------------------
DEFAULTS = {
    "samples": 10,
    "min_distance": 100.0,
    "max_distance": 400.0,
    "hemisphere": "vertical",       # "vertical" or "horizontal"
    "placement": None,              # None = no randomization
    "negative_ratio": 0.1,
    "val_split": 0.2,
    "enable_jitter": True,
    "jitter_max_pitch": 5.0,
    "jitter_max_yaw": 5.0,
}


# ---------------------------------------------------------------------------
# Per-object definitions
# ---------------------------------------------------------------------------
# Each key is the object's canonical name (used in GENERATE lists, folder
# names, and as the YOLO class name).
#
# "actor_label" must match the actor's Label in the UE5 scene.
# If omitted, it defaults to the canonical name (the dict key).
#
# "camera_group" determines the output sub-folder and the default hemisphere.
#
# "placement" — set to a dict to enable per-frame randomization:
#   {
#       "xy_range_x": float,    # max XY offset (cm) from original position
#       "xy_range_y": float,
#       "yaw_range": float,     # degrees — uniform [0, range]
#       "roll_range": float,    # degrees — uniform [-range, range]
#       "pitch_min": float,     # degrees
#       "pitch_max": float,     # degrees
#   }
# ---------------------------------------------------------------------------

OBJECT_DEFS = {
    # =========================================================================
    # cam_front objects — horizontal hemisphere (orbits around at eye level)
    # =========================================================================
    "gate_sawfish": {
        "camera_group": "cam_front",
        "hemisphere": "horizontal",
        "samples": 0,
        "min_distance": 150.0,
        "max_distance": 500.0,
        "co_visible": ["gate_shark"],
        "theta_range": (105.0, 255.0),  # gate faces -X; avoid side-on skeleton occlusion
        "keep_visible": ["gate"],  # HideInNegative actor labels to keep visible
    },
    "gate_shark": {
        "camera_group": "cam_front",
        "hemisphere": "horizontal",
        "samples": 0,
        "min_distance": 150.0,
        "max_distance": 500.0,
        "co_visible": ["gate_sawfish"],
        "theta_range": (105.0, 255.0),  # gate faces -X; avoid side-on skeleton occlusion
        "keep_visible": ["gate"],  # HideInNegative actor labels to keep visible
    },
    "red_pipe": {
        "camera_group": "cam_front",
        "hemisphere": "horizontal",
        "samples": 0,
        "min_distance": 100.0,
        "max_distance": 400.0,
    },
    "white_pipe": {
        "camera_group": "cam_front",
        "hemisphere": "horizontal",
        "samples": 0,
        "min_distance": 100.0,
        "max_distance": 400.0,
    },
    "torpedo_map": {
        "camera_group": "cam_front",
        "hemisphere": "horizontal",
        "samples": 0,
        "min_distance": 100.0,
        "max_distance": 400.0,
    },
    "bin_whole": {
        "camera_group": "cam_front",
        "hemisphere": "horizontal",
        "samples": 0,
        "min_distance": 200.0,
        "max_distance": 600.0,
        "keep_visible": ["bin_shark", "bin_sawfish"],
    },
    "octagon": {
        "camera_group": "cam_front",
        "hemisphere": "horizontal",
        "samples": 0,
        "min_distance": 150.0,
        "max_distance": 600.0,
        "keep_visible": ["octagon_masa"],
    },

    # =========================================================================
    # cam_bottom objects — vertical hemisphere (orbits above, bird's-eye)
    # =========================================================================
    "bin_shark": {
        "camera_group": "cam_bottom",
        "hemisphere": "vertical",
        "samples": 0,
        "min_distance": 80.0,
        "max_distance": 250.0,
        "phi_max": 30.0,
        "co_visible": ["bin_sawfish", "bin_whole"],
        "keep_visible": ["bin_sawfish", "bin_whole"],  # HideInNegative actor labels to keep visible
    },
    "bin_sawfish": {
        "camera_group": "cam_bottom",
        "hemisphere": "vertical",
        "samples": 0,
        "min_distance": 80.0,
        "max_distance": 250.0,
        "phi_max": 30.0,
        "keep_visible": ["bin_shark", "bin_whole"],
        "co_visible": ["bin_shark", "bin_whole"],
    },
    "octagon_table": {
        "camera_group": "cam_bottom",
        "hemisphere": "vertical",
        "samples": 0,
        "min_distance": 100.0,
        "max_distance": 300.0,
        "phi_max": 60.0,  # cut equator band — prevent front-on camera alignment
        "keep_visible": ["octagon_masa"],
    },

    # =========================================================================
    # cam_bottom_seg objects — vertical hemisphere + on-table randomization
    # =========================================================================
    "bottle": {
        "camera_group": "cam_bottom_seg",
        "hemisphere": "vertical",
        "samples": 0,
        "min_distance": 60.0,
        "max_distance": 250.0,
        "placement": {
            "xy_range_x": 18.0,
            "xy_range_y": 15.0,
            "yaw_range": 360.0,
            "roll_range": 90.0,
            "pitch_min": -90.0,
            "pitch_max": 0.0,
        },
    },
}


# ---------------------------------------------------------------------------
# Camera groups — maps group name to list of object names
# ---------------------------------------------------------------------------
CAMERA_GROUPS = {
    "cam_front": [
        "gate_sawfish", "gate_shark", "red_pipe", "white_pipe",
        "torpedo_map", "bin_whole", "octagon",
    ],
    "cam_bottom": [
        "bin_shark", "bin_sawfish", "octagon_table",
    ],
    "cam_bottom_seg": [
        "bottle",
    ],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_object_config(name):
    """Return a merged config dict for the given object (defaults + overrides).

    Args:
        name: Canonical object name (key in OBJECT_DEFS).

    Returns:
        dict with all config fields populated.

    Raises:
        KeyError: If the object name is not in the registry.
    """
    if name not in OBJECT_DEFS:
        raise KeyError(
            f"Object '{name}' not found in registry. "
            f"Valid names: {sorted(OBJECT_DEFS.keys())}"
        )

    obj = OBJECT_DEFS[name]
    merged = dict(DEFAULTS)
    merged.update(obj)

    # samples=0 means "use default"
    if merged.get("samples", 0) == 0:
        merged["samples"] = DEFAULTS["samples"]

    # Default actor_label to the canonical name if not specified
    if "actor_label" not in merged:
        merged["actor_label"] = name

    return merged


def resolve_targets(selection_list):
    """Expand a GENERATE selection list into individual object names.

    Supports:
        ["all"]                          → every object in the registry
        ["cam_front"]                    → all objects in the cam_front group
        ["cam_front", "bottle"]          → cam_front group + bottle
        ["gate_sawfish", "octagon"]      → just those two

    Args:
        selection_list: List of strings — object names, group names, or "all".

    Returns:
        Deduplicated list of canonical object names in registry order.

    Raises:
        ValueError: If a name is not a valid object or group.
    """
    result = []

    for item in selection_list:
        if item == "all":
            result.extend(OBJECT_DEFS.keys())
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

    # Deduplicate while preserving registry order
    seen = set()
    ordered = []
    for name in result:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    return ordered


def get_all_class_names():
    """Return all canonical object names in sorted order.

    This sorted order defines the canonical class ID assignment when merging
    multiple per-object datasets into a single multi-class dataset.
    """
    return sorted(OBJECT_DEFS.keys())
