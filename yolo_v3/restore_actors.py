"""
restore_actors.py — Emergency actor position recovery.

Run this inside the UE5 Editor Python console when actors are stuck
underground after a failed or interrupted dataset generation run.

It reads actor_restore.json (written by generate_yolo_v3.py before
moving actors underground) and teleports each listed actor back to
its original position and rotation.

Usage:
    py "C:/Users/deniz/ws/UE5_SyntheticDataGen/yolo_v3/restore_actors.py"
"""

import json
import os

import unreal

SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "actor_restore.json")


def restore_actors_from_snapshot():
    if not os.path.exists(SNAPSHOT_PATH):
        unreal.log_warning(f"No restore snapshot found at: {SNAPSHOT_PATH}")
        unreal.log_warning("Nothing to restore — actors may already be in their correct positions.")
        return

    with open(SNAPSHOT_PATH) as f:
        snapshot = json.load(f)

    if not snapshot:
        unreal.log_warning("Restore snapshot is empty.")
        return

    subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    all_actors = subsys.get_all_level_actors()
    actor_by_label = {a.get_actor_label(): a for a in all_actors}

    restored = 0
    missing = []
    for label, transform in snapshot.items():
        actor = actor_by_label.get(label)
        if actor is None:
            missing.append(label)
            continue
        loc = unreal.Vector(transform["x"], transform["y"], transform["z"])
        rot = unreal.Rotator(transform["roll"], transform["pitch"], transform["yaw"])
        actor.set_actor_location_and_rotation(loc, rot, sweep=False, teleport=True)
        restored += 1

    unreal.log("=" * 60)
    unreal.log(f"ACTOR RESTORE COMPLETE: {restored}/{len(snapshot)} actors restored")
    if missing:
        unreal.log_warning(f"  Not found in scene: {missing}")
    unreal.log("=" * 60)

    os.remove(SNAPSHOT_PATH)
    unreal.log(f"  Snapshot deleted: {SNAPSHOT_PATH}")


restore_actors_from_snapshot()
