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
#   Individual: "gate_searchrescue", "gate_surveyrepair", "red_pipe", "white_pipe",
#               "torpedo_map", "torpedo_hole", "bin_whole", "octagon",
#               "slalom",
#               "bin_shark", "bin_sawfish", "octagon_table",
#               "bottle",
#               "electric_link", "bandaid_link", "nutbolt_link", "pill_link"
#   Groups:     "cam_front", "cam_bottom", "cam_bottom_seg"
#   Special:    "all"
#   Helpers (co_visible only, not generated standalone):
#               "slalom_white_pipe"
# =============================================================================


# ---------------------------------------------------------------------------
# Default values — applied to any field not specified per-object
# ---------------------------------------------------------------------------
DEFAULTS = {
    "samples": 50,
    "min_distance": 100.0,
    "max_distance": 400.0,
    "hemisphere": "vertical",       # "vertical" or "horizontal"
    "placement": None,              # None = no randomization
    "negative_ratio": 0.1,
    "val_split": 0.2,
    "enable_jitter": True,
    "jitter_max_pitch": 5.0,
    "jitter_max_yaw": 5.0,
    "samples_on_edges": True,       # False = hide actors whose bbox intersects image edge
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
# "orbit_anchor_label" can point the camera orbit/aim center at a separate
# scene actor while keeping actor_label as the labeled object. Use this for
# multi-object setups where the camera should frame the whole setup.
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
#       "slot_group": str,      # optional shared placement group name
#       "slot_grid": (int, int),# optional (cols, rows) shuffled each frame
#       "slot_margin_x": float, # optional inset inside each slot (cm)
#       "slot_margin_y": float, # optional inset inside each slot (cm)
#   }
#
# "placement_collision" — set to False for visible support props (for example
#   a table or basket) that should remain in the scene but should NOT repel
#   placement-randomized objects in XY collision checks.
#
# "hard_negative_actors" — list of HideInNegative actor labels that should
#   appear ONLY in negative frames (underground in positives). Teaches the
#   model that these shapes are not the target class.
#
# "keep_visible_unlabeled" — list of canonical object names (registry keys)
#   whose actors remain visible in positive frames but are NOT labeled (no
#   bbox/mask contribution). Use when other TrainObject classes must share
#   the scene with the target (e.g., link props next to octagon_table) and
#   avoiding false negatives is guaranteed because those classes are never
#   trained in the same model. Behavior:
#     * Per positive frame, each listed object's OWN `placement` and/or
#       `rotation_dr` is applied (same sampler used when it is the target).
#     * If the listed object has neither `placement` nor `rotation_dr`, the
#       actor stays at its original scene pose every frame (no motion).
#     * When multiple placement-enabled entries share the same `slot_group`,
#       they are assigned shuffled `slot_grid` cells inside one shared XY area
#       before collision rejection runs. This avoids repeated intersections
#       without baking a fixed object-to-quadrant bias into the dataset.
#     * Placement-enabled entries participate in the same XY collision-
#       rejection loop as the target and `co_visible` (using
#       YOLO_V3_OBJECT_MIN_SEPARATION and YOLO_V3_MAX_COLLISION_RETRIES), so
#       they don't overlap each other or the target.
#     * In negative frames, all listed actors go underground.
#
# --- Visibility flags — which one do I need? -------------------------------
# Four flags control non-target actor visibility. They target different
# actor categories and produce different label outcomes:
#
#   Flag                      | Actor category       | Visible? | Labeled?
#   --------------------------+----------------------+----------+---------
#   co_visible                | TrainObject          | positive | YES (own class)
#   sub_actors                | TrainObject/other    | positive | YES (target's class)
#   keep_visible              | HideInNegative only  | positive | NO
#   keep_visible_unlabeled    | TrainObject          | positive | NO
#   hard_negative_actors      | HideInNegative only  | negative | NO
#
# Quick decision guide:
#   - "I want this labeled as a different class" → co_visible
#   - "I want this labeled as the same class as the target" → sub_actors
#   - "It's a skeleton/body/table mesh (HideInNegative tag), keep it in frame"
#       → keep_visible
#   - "It's a TrainObject from a different model, keep it in frame but don't
#      label it" → keep_visible_unlabeled
#      (inherits that object's placement/rotation_dr for motion; static if
#       neither is set)
#   - "It's a confusing shape (HideInNegative), show it only in negatives so
#      the model learns it's NOT the target" → hard_negative_actors
# ---------------------------------------------------------------------------
#
# "samples_on_edges" — set to False to hide co-visible/sub_actors whose
#   bounding box intersects an image edge for that frame. The actor is
#   keyframed underground so it disappears from both the rendered image
#   and the labels. Default: True (keep partial objects).
#
# "rotation_dr" — set to a dict to enable per-frame rotational sway:
#   {
#       "mode": "bottom_pivot",     # keep the actor's bottom point fixed
#       "roll_range": float,        # degrees — uniform [-range, range]
#       "pitch_range": float,       # degrees — uniform [-range, range]
#       "apply_to_self": bool,      # apply to actor_label (default True)
#       "apply_to_sub_actors": bool # apply to sub_actors (default False)
#   }
#
# "variant_tags" — list of UE5 actor tags (e.g. ["ver1", "ver2"]) for per-frame
#   variant alternation. Per positive frame, exactly one tag is "active":
#   actors with the active tag stay above ground, all others go underground.
#   Index alternates deterministically (i % N) for an exact even split.
#   In negative frames, all variant actors go underground.
#
# "variant_role" — required when variant_tags is set. One of:
#   "sub_actor_owner": TrainObject-tagged variant actors become sub_actors of
#       this class (each gets its own bbox label). Use exactly once across the
#       target + co_visible chain.
#   "visual_only": variant actors are visual (no bbox label for this class).
#       The class bbox comes from this object's actor_label (e.g. a skeleton).
# ---------------------------------------------------------------------------

OBJECT_DEFS = {
    # =========================================================================
    # cam_front objects — horizontal hemisphere (orbits around at eye level)
    # =========================================================================
    "gate_searchrescue": {
        "camera_group": "cam_front",
        "class_id": 0,
        "hemisphere": "horizontal",
        "samples": 2500,
        "min_distance": 250.0,
        "max_distance": 500.0,
        "orbit_anchor_label": "gate",
        "co_visible": ["gate_surveyrepair"],
        "theta_range": (120.0, 240.0),  # gate faces -X; avoid side-on skeleton occlusion
        "keep_visible": ["gate"],  # HideInNegative actor labels to keep visible
    },
    "gate_surveyrepair": {
        "camera_group": "cam_front",
        "class_id": 1,
        "hemisphere": "horizontal",
        "samples": 25,
        "min_distance": 250.0,
        "max_distance": 500.0,
        "orbit_anchor_label": "gate",
        "co_visible": ["gate_searchrescue"],
        "theta_range": (120.0, 240.0),  # gate faces -X; avoid side-on skeleton occlusion
        "keep_visible": ["gate"],  # HideInNegative actor labels to keep visible
    },
    "red_pipe": {
        "camera_group": "cam_front",
        "class_id": 2,
        "hemisphere": "horizontal",
        "samples": 0,
        "min_distance": 100.0,
        "max_distance": 400.0,
        #"hard_negative_actors": ["gate"],  # gate frame visible only in negatives to prevent pole/gate confusion
    },
    "white_pipe": {
        "camera_group": "cam_front",
        "class_id": 3,
        "hemisphere": "horizontal",
        "samples": 0,
        "min_distance": 100.0,
        "max_distance": 400.0,
        #"hard_negative_actors": ["gate"],  # gate frame visible only in negatives to prevent pole/gate confusion
    },
    "torpedo_map": {
        "camera_group": "cam_front",
        "class_id": 4,
        "hemisphere": "horizontal",
        "samples": 3000,
        "min_distance": 100.0,
        "max_distance": 300.0,
        "theta_range": (120.0, 240.0),
        "co_visible": ["torpedo_hole"],
        "keep_visible": ["torpedo_mesh"],
        # 50/50 variant split — actors tagged ver1/ver2 swap visibility per frame.
        # variant_role="visual_only": this object's variant actors are visual (map images),
        # never get a bbox label. The skeleton (actor_label="torpedo_map") provides the bbox.
        "variant_tags": ["ver1", "ver2"],
        "variant_role": "visual_only",
        # Disable global occlusion + min-bbox filters: torpedo_map's bbox geometrically
        # encloses every hole, so refine-mode incorrectly flags it as "occluded behind"
        # the nearer holes; the refined silhouette of the thin skeleton then drops below
        # the min-size floor and is silently discarded.
        "apply_occlusion_filter": False,
        "apply_min_bbox_filter": False,
    },
    "torpedo_hole": {
        "actor_label": "torpedo_hole_center",  # anchor actor at geometric center of holes
        "camera_group": "cam_front",
        "class_id": 5,
        "hemisphere": "horizontal",
        "samples": 3000,
        "min_distance": 100.0,
        "max_distance": 300.0,
        "theta_range": (120.0, 240.0),
        "co_visible": ["torpedo_map"],
        "keep_visible": ["torpedo_mesh"],
        "skip_target_bbox": True,           # anchor — no bbox for itself
        # 50/50 variant split — 4 hole boxes per version, swap per frame.
        # variant_role="sub_actor_owner": TrainObject-tagged variant actors become
        # sub_actors of THIS class (each hole gets its own bbox).
        "variant_tags": ["ver1", "ver2"],
        "variant_role": "sub_actor_owner",
        # Far holes on angled-board views project below the global 4x8 px floor and
        # also get cross-flagged by refine-mode occlusion against nearer holes.
        "apply_occlusion_filter": False,
        "apply_min_bbox_filter": False,
    },
    "bin_whole": {
        "camera_group": "cam_front",
        "class_id": 6,
        "hemisphere": "horizontal",
        "samples": 0,
        "min_distance": 200.0,
        "max_distance": 600.0,
        "keep_visible": ["bin_shark", "bin_sawfish"],
    },
    "octagon": {
        "camera_group": "cam_front",
        "class_id": 7,
        "hemisphere": "horizontal",
        "samples": 2500,
        "min_distance": 150.0,
        "max_distance": 600.0,
        "keep_visible": [
            "octagon_masa", "octagon_basket", "octagon_basket2",
            "electric_link_body", "bandaid_link_body",
            "nutbolt_link_body", "pill_link_body", "basket_redcross_segment_link",
            "basket_warning_segment_link"
        ],
        # Link props share the scene visually but live in a different model —
        # keep them rendered so the table is shown in context without
        # producing false-negative labels for the table class.
        "keep_visible_unlabeled": [
            "electric_link", "bandaid_link", "nutbolt_link", "pill_link",
            "basket_redcross_segment_link", "basket_warning_segment_link",
        ],
    },
    "slalom": {
        "actor_label": "slalom_center",
        "class_name": "red_pipe",         # merged class name matches standalone red_pipe
        "camera_group": "cam_front",
        "class_id": 8,
        "hemisphere": "horizontal",
        "samples": 2500,
        "min_distance": 500.0,
        "max_distance": 1500.0,
        "skip_target_bbox": True,         # anchor actor — no bbox for itself
        "samples_on_edges": False,        # hide partially visible pipes at image edges
        "co_visible": ["slalom_white_pipe"],
        "sub_actors": ["red_pipe", "red_pipe_2", "red_pipe_3"],
        #"hard_negative_actors": ["gate"],  # gate frame visible only in negatives to prevent pole/gate confusion
        "rotation_dr": {
            "mode": "bottom_pivot",
            "roll_range": 8.0,
            "pitch_range": 8.0,
            "apply_to_self": False,
            "apply_to_sub_actors": True,
        },
    },
    "slalom_white_pipe": {
        "actor_label": "white_pipe",
        "class_name": "white_pipe",       # merged class name matches standalone white_pipe
        "camera_group": "cam_front",
        "class_id": 9,
        "hemisphere": "horizontal",
        "samples": 0,
        "helper": True,                   # co_visible helper — not generated standalone
        "sub_actors": ["white_pipe_2", "white_pipe_3", "white_pipe_4",
                       "white_pipe_5", "white_pipe_6"],
        "rotation_dr": {
            "mode": "bottom_pivot",
            "roll_range": 8.0,
            "pitch_range": 8.0,
            "apply_to_self": True,
            "apply_to_sub_actors": True,
        },
    },

    # =========================================================================
    # cam_bottom objects — vertical hemisphere (orbits above, bird's-eye)
    # =========================================================================
    "bin_shark": {
        "camera_group": "cam_bottom",
        "class_id": 0,
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
        "class_id": 1,
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
        "class_id": 2,
        "hemisphere": "vertical",
        "samples": 5000,
        "min_distance": 100.0,
        "max_distance": 300.0,
        "phi_max": 40.0,  # cut equator band — prevent front-on camera alignment
        "keep_visible": [
            "octagon_masa",
            "electric_link_body", "bandaid_link_body",
            "nutbolt_link_body", "pill_link_body",
        ],
        # Link props share the scene visually but live in a different model —
        # keep them rendered so the table is shown in context without
        # producing false-negative labels for the table class.
        "keep_visible_unlabeled": [
            "electric_link", "bandaid_link", "nutbolt_link", "pill_link",
        ],
        "placement_collision": False,
    },

    # =========================================================================
    # cam_bottom_seg objects — vertical hemisphere + on-table randomization
    # =========================================================================
    "bottle": {
        "camera_group": "cam_bottom_seg",
        "class_id": 0,
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

    # Octagon table link objects — all 4 appear together per frame, each
    # randomized on the table with collision rejection. Fixed tilt (flat on
    # table), yaw free. Actor labels match class names.
    "electric_link": {
        "camera_group": "cam_bottom_seg",
        "class_id": 0,
        "hemisphere": "vertical",
        "samples": 3,
        "min_distance": 125.0,
        "max_distance": 250.0,
        "phi_max": 20.0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        "co_visible": ["bandaid_link", "nutbolt_link", "pill_link",
                       "basket_redcross_segment_link",
                       "basket_warning_segment_link",
                       "octagon_table_segment_link"],
        "keep_visible": ["electric_link_body", "bandaid_link_body",
                         "nutbolt_link_body", "pill_link_body",
                         "octagon_masa",
                         "octagon_basket", "octagon_basket2"],
        "placement": {
            "xy_range_x": 18.0,
            "xy_range_y": 15.0,
            "yaw_range": 360.0,
            "roll_range": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0,
            "slot_group": "octagon_table_links",
            "slot_grid": (2, 2),
            "slot_margin_x": 2.0,
            "slot_margin_y": 2.0,
        },
    },
    "bandaid_link": {
        "camera_group": "cam_bottom_seg",
        "class_id": 1,
        "hemisphere": "vertical",
        "samples": 15,
        "min_distance": 60.0,
        "max_distance": 250.0,
        "phi_max": 20.0,
        "co_visible": ["electric_link", "nutbolt_link", "pill_link",
                       "basket_redcross_segment_link",
                       "basket_warning_segment_link",
                       "octagon_table_segment_link"],
        "keep_visible": ["electric_link_body", "bandaid_link_body",
                         "nutbolt_link_body", "pill_link_body",
                         "octagon_masa",
                         "octagon_basket", "octagon_basket2"],
        "placement": {
            "xy_range_x": 18.0,
            "xy_range_y": 15.0,
            "yaw_range": 360.0,
            "roll_range": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0,
            "slot_group": "octagon_table_links",
            "slot_grid": (2, 2),
            "slot_margin_x": 2.0,
            "slot_margin_y": 2.0,
        },
    },
    "nutbolt_link": {
        "camera_group": "cam_bottom_seg",
        "class_id": 2,
        "hemisphere": "vertical",
        "samples": 15,
        "min_distance": 60.0,
        "max_distance": 250.0,
        "phi_max": 20.0,
        "co_visible": ["electric_link", "bandaid_link", "pill_link",
                       "basket_redcross_segment_link",
                       "basket_warning_segment_link",
                       "octagon_table_segment_link"],
        "keep_visible": ["electric_link_body", "bandaid_link_body",
                         "nutbolt_link_body", "pill_link_body",
                         "octagon_masa",
                         "octagon_basket", "octagon_basket2"],
        "placement": {
            "xy_range_x": 18.0,
            "xy_range_y": 15.0,
            "yaw_range": 360.0,
            "roll_range": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0,
            "slot_group": "octagon_table_links",
            "slot_grid": (2, 2),
            "slot_margin_x": 2.0,
            "slot_margin_y": 2.0,
        },
    },
    "pill_link": {
        "camera_group": "cam_bottom_seg",
        "class_id": 3,
        "hemisphere": "vertical",
        "samples": 15,
        "min_distance": 60.0,
        "max_distance": 250.0,
        "phi_max": 20.0,
        "co_visible": ["electric_link", "bandaid_link", "nutbolt_link",
                       "basket_redcross_segment_link",
                       "basket_warning_segment_link",
                       "octagon_table_segment_link"],
        "keep_visible": ["electric_link_body", "bandaid_link_body",
                         "nutbolt_link_body", "pill_link_body",
                         "octagon_masa",
                         "octagon_basket", "octagon_basket2"],
        "placement": {
            "xy_range_x": 18.0,
            "xy_range_y": 15.0,
            "yaw_range": 360.0,
            "roll_range": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0,
            "slot_group": "octagon_table_links",
            "slot_grid": (2, 2),
            "slot_margin_x": 2.0,
            "slot_margin_y": 2.0,
        },
    },

    # Static seg classes — single actors, no placement randomization. They
    # live in the scene at fixed poses and get labeled whenever any
    # cam_bottom_seg target is rendered (co_visible with all link classes).
    "basket_redcross_segment_link": {
        "camera_group": "cam_bottom_seg",
        "class_id": 4,
        "hemisphere": "vertical",
        "samples": 15,
        "min_distance": 110.0,
        "max_distance": 300.0,
        "phi_max": 20.0,
        "co_visible": ["electric_link", "bandaid_link", "nutbolt_link", "pill_link",
                       "basket_warning_segment_link",
                       "octagon_table_segment_link"],
        "keep_visible": ["electric_link_body", "bandaid_link_body",
                         "nutbolt_link_body", "pill_link_body",
                         "octagon_masa",
                         "octagon_basket", "octagon_basket2"],
        "placement_collision": False,
    },
    "basket_warning_segment_link": {
        "camera_group": "cam_bottom_seg",
        "class_id": 5,
        "hemisphere": "vertical",
        "samples": 15,
        "min_distance": 150.0,
        "max_distance": 350.0,
        "phi_max": 20.0,
        "co_visible": ["electric_link", "bandaid_link", "nutbolt_link", "pill_link",
                       "basket_redcross_segment_link",
                       "octagon_table_segment_link"],
        "keep_visible": ["electric_link_body", "bandaid_link_body",
                         "nutbolt_link_body", "pill_link_body",
                         "octagon_masa",
                         "octagon_basket", "octagon_basket2"],
        "placement_collision": False,
    },
    "octagon_table_segment_link": {
        "camera_group": "cam_bottom_seg",
        "class_id": 6,
        "hemisphere": "vertical",
        "samples": 15,
        "min_distance": 110.0,
        "max_distance": 300.0,
        "phi_max": 20.0,
        "co_visible": ["electric_link", "bandaid_link", "nutbolt_link", "pill_link",
                       "basket_redcross_segment_link",
                       "basket_warning_segment_link"],
        "keep_visible": ["electric_link_body", "bandaid_link_body",
                         "nutbolt_link_body", "pill_link_body",
                         "octagon_masa",
                         "octagon_basket", "octagon_basket2"],
        "placement_collision": False,
        # Hide the same-color table mesh during both SceneCapture passes so
        # the differential mask isn't washed out to zero.
        "mask_occluder": ["octagon_masa"],
    },
}


# ---------------------------------------------------------------------------
# Camera groups — maps group name to list of object names
# ---------------------------------------------------------------------------
CAMERA_GROUPS = {
    "cam_front": [
        "gate_searchrescue", "gate_surveyrepair", "red_pipe", "white_pipe",
        "torpedo_map", "torpedo_hole_disabled", "bin_whole_TODO", "octagon",
        "slalom",
    ],
    "cam_bottom": [
        "bin_shark", "bin_sawfish", "octagon_table",
    ],
    "cam_bottom_seg": [
        "bottle",
        "electric_link", "bandaid_link", "nutbolt_link", "pill_link",
        "basket_redcross_segment_link",
        "basket_warning_segment_link",
        "octagon_table_segment_link",
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
        ["gate_searchrescue", "octagon"]      → just those two

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
