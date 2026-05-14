"""
Remove one or more classes from YOLO label files.

By default this script compacts the remaining class IDs and rewrites data.yaml,
which is what most YOLO training tools expect. Use --keep-class-ids if you
only want to delete matching rows and leave all other IDs unchanged.

Examples:
    python strip_yolo_class.py --source C:/merged --output C:/merged_no_gate --class-name gate_searchrescue
    python strip_yolo_class.py --source C:/merged --in-place --class-id 3 --dry-run
    python strip_yolo_class.py --source C:/merged --output C:/merged_no_3_5 --class-id 3 --class-id 5 --drop-empty-images
"""

import argparse
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def parse_data_yaml(yaml_path):
    """Parse the simple data.yaml format written by this repo and Ultralytics."""
    meta = {}
    names = {}
    if not yaml_path.exists():
        return meta, names

    parsing_names = False
    list_index = 0
    with yaml_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("names:"):
                parsing_names = True
                value = stripped.split(":", 1)[1].strip()
                if value.startswith("[") and value.endswith("]"):
                    entries = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
                    names = {idx: name for idx, name in enumerate(entries)}
                    parsing_names = False
                continue

            if parsing_names:
                if stripped.startswith("-"):
                    names[list_index] = stripped[1:].strip().strip("'\"")
                    list_index += 1
                    continue
                if ":" in stripped:
                    key, value = stripped.split(":", 1)
                    try:
                        names[int(key.strip())] = value.strip().strip("'\"")
                        continue
                    except ValueError:
                        parsing_names = False
                else:
                    parsing_names = False

            if ":" in stripped and not parsing_names:
                key, value = stripped.split(":", 1)
                meta[key.strip()] = value.strip()

    return meta, names


def write_data_yaml(yaml_path, meta, names):
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8", newline="\n") as f:
        for key in ("path", "train", "val", "test", "task"):
            if key in meta:
                f.write(f"{key}: {meta[key]}\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for class_id in sorted(names):
            f.write(f"  {class_id}: {names[class_id]}\n")


def collect_label_files(root):
    label_files = [p for p in root.rglob("*.txt") if p.parent.name.lower() == "labels"]
    if label_files:
        return sorted(label_files)
    return sorted(root.rglob("*.txt"))


def collect_observed_ids(label_files):
    ids = set()
    for label_file in label_files:
        with label_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    ids.add(int(parts[0]))
                except ValueError:
                    continue
    return ids


def resolve_remove_ids(class_ids, class_names, names):
    remove_ids = set(class_ids or [])
    if class_names:
        if not names:
            raise ValueError("--class-name requires data.yaml with names")
        name_to_ids = {}
        for class_id, name in names.items():
            name_to_ids.setdefault(name, []).append(class_id)
        for class_name in class_names:
            if class_name not in name_to_ids:
                raise ValueError(f"Class name not found in data.yaml: {class_name}")
            remove_ids.update(name_to_ids[class_name])
    if not remove_ids:
        raise ValueError("Provide at least one --class-id or --class-name")
    return remove_ids


def build_id_map(observed_ids, names, remove_ids, keep_class_ids):
    remaining_ids = sorted((set(observed_ids) | set(names)) - remove_ids)
    if keep_class_ids:
        return {class_id: class_id for class_id in remaining_ids}
    return {old_id: new_id for new_id, old_id in enumerate(remaining_ids)}


def rewrite_label_file(label_file, remove_ids, id_map, dry_run):
    kept_lines = []
    removed = 0
    remapped = 0
    malformed = 0

    with label_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            try:
                old_id = int(parts[0])
            except ValueError:
                kept_lines.append(stripped)
                malformed += 1
                continue

            if old_id in remove_ids:
                removed += 1
                continue
            new_id = id_map.get(old_id, old_id)
            if new_id != old_id:
                remapped += 1
            parts[0] = str(new_id)
            kept_lines.append(" ".join(parts))

    if not dry_run:
        with label_file.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))

    return removed, remapped, malformed, len(kept_lines)


def find_matching_images(label_file):
    if label_file.parent.name.lower() != "labels":
        return []
    image_dir = label_file.parent.parent / "images"
    if not image_dir.is_dir():
        return []
    matches = []
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{label_file.stem}{ext}"
        if candidate.exists():
            matches.append(candidate)
    return matches


def remove_empty_sample(label_file, dry_run):
    removed_images = 0
    for image_path in find_matching_images(label_file):
        if not dry_run:
            image_path.unlink()
        removed_images += 1
    if not dry_run and label_file.exists():
        label_file.unlink()
    return removed_images


def strip_dataset(source, output, in_place, class_ids, class_names, keep_class_ids,
                  drop_empty_images, dry_run):
    source_root = Path(source).resolve()
    output_root = Path(output).resolve() if output else None

    if not source_root.exists():
        raise FileNotFoundError(f"Source does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source must be a directory: {source_root}")
    if bool(output_root) == bool(in_place):
        raise ValueError("Choose exactly one: --output or --in-place")
    if output_root and output_root == source_root:
        raise ValueError("--output must be different from --source; use --in-place instead")

    work_root = source_root if in_place else output_root
    if output_root:
        if output_root.exists() and not dry_run:
            raise FileExistsError(f"Output already exists: {output_root}")
        if not dry_run:
            shutil.copytree(source_root, output_root)

    yaml_path = work_root / "data.yaml"
    source_yaml_path = source_root / "data.yaml"
    meta, names = parse_data_yaml(source_yaml_path)
    remove_ids = resolve_remove_ids(class_ids, class_names, names)

    label_files = collect_label_files(work_root if not dry_run or in_place else source_root)
    observed_ids = collect_observed_ids(label_files)
    id_map = build_id_map(observed_ids, names, remove_ids, keep_class_ids)

    print("=" * 60)
    print("YOLO LABEL STRIPPER")
    print("=" * 60)
    print(f"Source: {source_root}")
    print(f"Mode: {'in-place' if in_place else 'copy'}")
    if output_root:
        print(f"Output: {output_root}")
    print(f"Remove IDs: {sorted(remove_ids)}")
    print(f"Keep class IDs: {keep_class_ids}")
    print(f"Drop empty images: {drop_empty_images}")
    print(f"Label files: {len(label_files)}")
    if dry_run:
        print("Dry run: no files will be written")

    removed_lines = 0
    remapped_lines = 0
    malformed_lines = 0
    empty_labels = 0
    removed_images = 0

    for label_file in label_files:
        removed, remapped, malformed, kept = rewrite_label_file(
            label_file, remove_ids, id_map, dry_run
        )
        removed_lines += removed
        remapped_lines += remapped
        malformed_lines += malformed
        if kept == 0:
            empty_labels += 1
            if drop_empty_images:
                removed_images += remove_empty_sample(label_file, dry_run)

    if names and not keep_class_ids:
        new_names = {
            new_id: names.get(old_id, f"class_{old_id}")
            for old_id, new_id in id_map.items()
        }
        if not dry_run:
            write_data_yaml(yaml_path, meta, new_names)
    elif names and keep_class_ids:
        new_names = {
            class_id: name
            for class_id, name in names.items()
            if class_id not in remove_ids
        }
        if not dry_run:
            write_data_yaml(yaml_path, meta, new_names)

    print()
    print("Summary:")
    print(f"  Removed label rows: {removed_lines}")
    print(f"  Remapped label rows: {remapped_lines}")
    print(f"  Malformed rows kept: {malformed_lines}")
    print(f"  Empty label files after strip: {empty_labels}")
    if drop_empty_images:
        print(f"  Removed images for empty labels: {removed_images}")
    if names:
        print(f"  Remaining classes: {len(id_map)}")
        if not keep_class_ids:
            print(f"  ID map: {id_map}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove selected classes from YOLO labels and optionally compact class IDs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="YOLO dataset root")
    parser.add_argument("--output", default=None, help="Write stripped dataset copy to this folder")
    parser.add_argument("--in-place", action="store_true", help="Modify the source dataset directly")
    parser.add_argument("--class-id", type=int, action="append", default=[], help="Class ID to remove; repeatable")
    parser.add_argument("--class-name", action="append", default=[], help="Class name from data.yaml to remove; repeatable")
    parser.add_argument("--keep-class-ids", action="store_true", help="Delete rows but do not remap remaining IDs")
    parser.add_argument("--drop-empty-images", action="store_true",
                        help="Remove image+label samples that become empty after stripping")
    parser.add_argument("--dry-run", action="store_true", help="Preview work without writing files")
    args = parser.parse_args()

    strip_dataset(
        source=args.source,
        output=args.output,
        in_place=args.in_place,
        class_ids=args.class_id,
        class_names=args.class_name,
        keep_class_ids=args.keep_class_ids,
        drop_empty_images=args.drop_empty_images,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
