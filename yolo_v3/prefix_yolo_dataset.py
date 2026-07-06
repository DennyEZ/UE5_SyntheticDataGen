"""
Prefix YOLO dataset image and label filenames in place.

Drop this script into a generated or merged YOLO dataset folder and run it:

    python prefix_yolo_dataset.py

Or run it from anywhere:

    python prefix_yolo_dataset.py --source C:/merged_cam_front --prefix binSetupSyntheticJuly6_
    python prefix_yolo_dataset.py --source C:/merged_cam_front --prefix binSetupSyntheticJuly6_ --dry-run

The script renames files under YOLO split folders such as:

    train/images, train/labels, val/images, val/labels, test/images, test/labels

Labels and images are matched by filename stem in YOLO, so both sides receive
the same prefix. data.yaml does not need to change because it points at folders,
not individual files.
"""

import argparse
import os
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_EXTENSIONS = {".txt"}
DATASET_FILE_DIRS = {"images", "labels"}


def is_dataset_file(path):
    parent_name = path.parent.name.lower()
    suffix = path.suffix.lower()
    if parent_name == "images":
        return suffix in IMAGE_EXTENSIONS
    if parent_name == "labels":
        return suffix in LABEL_EXTENSIONS
    return False


def collect_dataset_files(root):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.parent.name.lower() in DATASET_FILE_DIRS
        and is_dataset_file(path)
    )


def build_rename_plan(root, prefix, include_already_prefixed=False):
    files = collect_dataset_files(root)
    plan = []
    skipped_prefixed = []

    for src in files:
        if src.name.startswith(prefix) and not include_already_prefixed:
            skipped_prefixed.append(src)
            continue
        dst = src.with_name(f"{prefix}{src.name}")
        if src != dst:
            plan.append((src, dst))

    return plan, skipped_prefixed


def validate_plan(plan):
    sources = {src.resolve() for src, _ in plan}
    targets = {}
    errors = []

    for src, dst in plan:
        resolved_dst = dst.resolve()
        if resolved_dst in targets:
            errors.append(f"Two files would rename to the same target: {targets[resolved_dst]} and {src} -> {dst}")
        targets[resolved_dst] = src

        if dst.exists() and resolved_dst not in sources:
            errors.append(f"Target already exists: {dst}")

    if errors:
        raise ValueError("Rename plan has collisions:\n  " + "\n  ".join(errors))


def temp_path_for(src):
    return src.with_name(f".__prefix_tmp__{os.getpid()}__{src.name}")


def apply_plan(plan, dry_run=False):
    if dry_run:
        return

    staged = []
    try:
        for src, _ in plan:
            tmp = temp_path_for(src)
            if tmp.exists():
                raise FileExistsError(f"Temporary file already exists: {tmp}")
            src.rename(tmp)
            staged.append((tmp, src))

        for (tmp, _src), (_original, dst) in zip(staged, plan):
            tmp.rename(dst)
    except Exception:
        for tmp, original in reversed(staged):
            if tmp.exists() and not original.exists():
                tmp.rename(original)
        raise


def prompt_prefix():
    while True:
        prefix = input("Prefix to add to all dataset images and labels: ").strip()
        if prefix:
            return prefix
        print("Prefix cannot be empty.")


def confirm(prompt):
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def prefix_dataset(source, prefix, include_already_prefixed=False, yes=False, dry_run=False):
    root = Path(source).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Source must be a folder: {root}")
    if not prefix:
        raise ValueError("Prefix cannot be empty")

    plan, skipped_prefixed = build_rename_plan(root, prefix, include_already_prefixed)
    validate_plan(plan)

    image_count = sum(1 for src, _ in plan if src.parent.name.lower() == "images")
    label_count = sum(1 for src, _ in plan if src.parent.name.lower() == "labels")

    print("=" * 60)
    print("YOLO DATASET PREFIXER")
    print("=" * 60)
    print(f"Dataset: {root}")
    print(f"Prefix: {prefix}")
    print(f"Images to rename: {image_count}")
    print(f"Labels to rename: {label_count}")
    print(f"Already prefixed skipped: {len(skipped_prefixed)}")
    if dry_run:
        print("Dry run: no files will be renamed")

    if not plan:
        print("\nNothing to rename.")
        return True

    print("\nExamples:")
    for src, dst in plan[:5]:
        print(f"  {src.relative_to(root)} -> {dst.relative_to(root)}")
    if len(plan) > 5:
        print(f"  ... {len(plan) - 5} more")

    if not dry_run and not yes and not confirm("\nRename these files now?"):
        print("Cancelled.")
        return False

    apply_plan(plan, dry_run=dry_run)

    print("\nSummary:")
    print(f"  Renamed images: {image_count}")
    print(f"  Renamed labels: {label_count}")
    print(f"  Total renamed: {len(plan)}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add a prefix to all YOLO dataset image and label filenames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default=".", help="YOLO dataset root; defaults to the current folder")
    parser.add_argument("--prefix", default=None, help="Prefix to add. If omitted, the script prompts for it")
    parser.add_argument("--include-already-prefixed", action="store_true",
                        help="Also prefix files that already start with this prefix")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Preview renames without writing files")
    args = parser.parse_args()

    prefix = args.prefix if args.prefix is not None else prompt_prefix()
    ok = prefix_dataset(
        source=args.source,
        prefix=prefix,
        include_already_prefixed=args.include_already_prefixed,
        yes=args.yes,
        dry_run=args.dry_run,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
