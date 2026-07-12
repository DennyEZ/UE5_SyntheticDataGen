"""
Octagon Link Merger — Merge the 4 cam_bottom_seg link datasets into one
multi-class dataset with universal short class names and fixed IDs.

Merges electric_link, bandaid_link, nutbolt_link and pill_link (each of which
also labels the co-visible basket/table segment classes) and remaps every
label to the universal class map:

    0: bandaid    1: electric   2: nutbolt   3: pill
    4: redcross   5: table      6: warning

Usage:
    python merge_links_seg.py --source_root C:/UE5_YOLO_Data_V3/ --output C:/merged_links/
    python merge_links_seg.py --source_root C:/UE5_YOLO_Data_V3/ --output C:/merged_links/ --dry_run

Output:
    output/
    ├── data.yaml           (universal names + fixed IDs, task: segment)
    ├── train/
    │   ├── images/         (prefixed: {source}_{original}.png)
    │   └── labels/         (remapped class IDs)
    └── val/
        ├── images/
        └── labels/

No UE5 dependency — runs with standard Python.
"""

import argparse
import os
import shutil
import sys

# Add script directory to path so we can import sibling modules
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from merge_datasets import iter_image_files, parse_data_yaml, remap_label_file

# Source dataset folders under {source_root}/cam_bottom_seg/
CAMERA_GROUP = "cam_bottom_seg"
LINK_SOURCES = ("bandaid_link", "electric_link", "nutbolt_link", "pill_link")

# Universal class map — fixed IDs, short names
UNIVERSAL_CLASSES = {
    0: "bandaid",
    1: "electric",
    2: "nutbolt",
    3: "pill",
    4: "redcross",
    5: "table",
    6: "warning",
}

# Long registry/scene class name → universal short name
NAME_MAP = {
    "bandaid_link": "bandaid",
    "electric_link": "electric",
    "nutbolt_link": "nutbolt",
    "pill_link": "pill",
    "basket_redcross_segment_link": "redcross",
    "octagon_table_segment_link": "table",
    "basket_warning_segment_link": "warning",
}

UNIVERSAL_ID = {name: cid for cid, name in UNIVERSAL_CLASSES.items()}


def build_source_remap(source_path):
    """Build {local_id: universal_id} for one source dataset from its data.yaml.

    Fails fast on class names that aren't in NAME_MAP — an unknown class means
    the source dataset is not one of the expected link datasets.
    """
    yaml_path = os.path.join(source_path, "data.yaml")
    local_classes, task = parse_data_yaml(yaml_path)
    if not local_classes:
        raise ValueError(f"No classes parsed from {yaml_path}")

    remap = {}
    for local_id, long_name in local_classes.items():
        short_name = NAME_MAP.get(long_name)
        if short_name is None:
            raise ValueError(
                f"Unknown class '{long_name}' in {yaml_path}. "
                f"Expected one of: {sorted(NAME_MAP.keys())}"
            )
        remap[local_id] = UNIVERSAL_ID[short_name]
    return remap, task


def merge_link_datasets(source_root, output_dir, dry_run=False):
    print("=" * 60)
    print("OCTAGON LINK DATASET MERGER (cam_bottom_seg)")
    print("=" * 60)

    group_dir = os.path.join(source_root, CAMERA_GROUP)

    # --- Step 1: Validate sources and build remaps ---
    source_info = []
    task_type = "segment"
    missing = []
    for obj_name in LINK_SOURCES:
        src_path = os.path.join(group_dir, obj_name)
        if not os.path.isdir(src_path):
            missing.append(src_path)
            continue
        remap, task = build_source_remap(src_path)
        if task:
            task_type = task
        source_info.append({"path": src_path, "obj_name": obj_name, "remap": remap})

    if missing:
        for path in missing:
            print(f"  WARNING: Source not found, skipping: {path}")
    if not source_info:
        print("ERROR: No source datasets found!")
        sys.exit(1)

    print(f"\nSources: {len(source_info)}")
    print(f"Task: {task_type}")
    print(f"Universal classes ({len(UNIVERSAL_CLASSES)}):")
    for cid in sorted(UNIVERSAL_CLASSES):
        print(f"  {cid}: {UNIVERSAL_CLASSES[cid]}")
    for info in source_info:
        print(f"\n  {info['obj_name']}: {info['path']}")
        print(f"    Local -> Universal: {info['remap']}")

    if dry_run:
        print("\n[DRY RUN] Would merge the above sources. No files written.")
        for split in ("train", "val"):
            total = sum(
                len(iter_image_files(os.path.join(info["path"], split, "images")))
                for info in source_info
                if os.path.isdir(os.path.join(info["path"], split, "images"))
            )
            print(f"  {split}: ~{total} images")
        return

    # --- Step 2: Fresh output tree ---
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, split, "images"))
        os.makedirs(os.path.join(output_dir, split, "labels"))

    # --- Step 3: Copy images + remap labels ---
    total_images = {"train": 0, "val": 0}
    total_labels = {"train": 0, "val": 0}

    for info in source_info:
        prefix = info["obj_name"]
        for split in ("train", "val"):
            src_img_dir = os.path.join(info["path"], split, "images")
            src_lbl_dir = os.path.join(info["path"], split, "labels")
            dst_img_dir = os.path.join(output_dir, split, "images")
            dst_lbl_dir = os.path.join(output_dir, split, "labels")
            if not os.path.isdir(src_img_dir):
                continue

            for img_path in iter_image_files(src_img_dir):
                basename = os.path.splitext(os.path.basename(img_path))[0]
                ext = os.path.splitext(img_path)[1].lower()
                new_basename = f"{prefix}_{basename}"

                shutil.copy2(img_path, os.path.join(dst_img_dir, f"{new_basename}{ext}"))
                total_images[split] += 1

                src_lbl = os.path.join(src_lbl_dir, f"{basename}.txt")
                if os.path.exists(src_lbl):
                    dst_lbl = os.path.join(dst_lbl_dir, f"{new_basename}.txt")
                    remap_label_file(src_lbl, dst_lbl, info["remap"])
                    total_labels[split] += 1

        print(f"  Merged: {prefix}")

    # --- Step 4: Unified data.yaml with universal names ---
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write(f"task: {task_type}\n")
        f.write(f"nc: {len(UNIVERSAL_CLASSES)}\n")
        f.write("names:\n")
        for cid in sorted(UNIVERSAL_CLASSES):
            f.write(f"  {cid}: {UNIVERSAL_CLASSES[cid]}\n")

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print(f"  Output: {output_dir}")
    print(f"  Classes: {len(UNIVERSAL_CLASSES)}")
    print(f"  Train: {total_images['train']} images, {total_labels['train']} labels")
    print(f"  Val:   {total_images['val']} images, {total_labels['val']} labels")
    print(f"  data.yaml: {yaml_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge the 4 cam_bottom_seg link datasets with universal class names.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_links_seg.py --source_root C:/UE5_YOLO_Data_V3/ --output C:/merged_links/
  python merge_links_seg.py --source_root C:/UE5_YOLO_Data_V3/ --output C:/merged_links/ --dry_run
        """)
    parser.add_argument("--source_root", type=str, required=True,
                        help="Root folder containing the cam_bottom_seg/ group folder")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the merged dataset (cleared first)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview merge without writing files")
    args = parser.parse_args()

    merge_link_datasets(args.source_root, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
