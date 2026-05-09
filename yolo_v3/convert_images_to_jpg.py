"""
Convert dataset images to JPG.

Works on YOLO datasets, merged datasets, and external image folders. If an
output folder is provided, the source tree is copied while image files are
written as .jpg. In-place mode converts images next to the originals and
removes the source image after a successful conversion unless --keep-originals
is used.

Examples:
    python convert_images_to_jpg.py --source C:/UE5_YOLO_Data_V3/cam_front/octagon --output C:/octagon_jpg
    python convert_images_to_jpg.py --source C:/merged_cam_front --in-place --quality 92
    python convert_images_to_jpg.py --source C:/external_dataset --output C:/external_dataset_jpg --dry-run
"""

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SOURCE_IMAGE_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".webp", ".jpg", ".jpeg"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}


def is_image_path(path):
    return path.suffix.lower() in SOURCE_IMAGE_EXTENSIONS


def resolve_output_path(src_file, source_root, output_root):
    rel_path = src_file.relative_to(source_root)
    return (output_root / rel_path).with_suffix(".jpg")


def image_to_rgb(image, background):
    if image.mode in {"RGBA", "LA"} or ("transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, background + (255,))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image.convert("RGB")


def convert_one(src_file, dst_file, quality, background, dry_run):
    if dry_run:
        return True

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    write_file = dst_file
    temp_file = None
    if src_file.resolve() == dst_file.resolve():
        temp_file = dst_file.with_name(f"{dst_file.stem}.__tmp_jpg__{dst_file.suffix}")
        write_file = temp_file

    try:
        with Image.open(src_file) as image:
            rgb = image_to_rgb(image, background)
            rgb.save(
                write_file,
                "JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
        if temp_file:
            os.replace(temp_file, dst_file)
        return True
    except (OSError, UnidentifiedImageError) as exc:
        print(f"  WARNING: could not convert {src_file}: {exc}")
        if temp_file and temp_file.exists():
            temp_file.unlink()
        return False


def copy_non_image(src_file, dst_file, dry_run):
    if dry_run:
        return
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)


def convert_dataset(source, output, in_place, quality, keep_originals, reencode_jpegs, overwrite, background, dry_run):
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

    files = [p for p in source_root.rglob("*") if p.is_file()]
    image_files = [p for p in files if is_image_path(p)]
    non_image_files = [p for p in files if not is_image_path(p)]

    print("=" * 60)
    print("DATASET JPG CONVERTER")
    print("=" * 60)
    print(f"Source: {source_root}")
    print(f"Mode: {'in-place' if in_place else 'copy'}")
    if output_root:
        print(f"Output: {output_root}")
    print(f"Quality: {quality}")
    print(f"Images found: {len(image_files)}")
    print(f"Other files: {len(non_image_files)}")
    if dry_run:
        print("Dry run: no files will be written")

    converted = 0
    skipped = 0
    removed = 0
    copied = 0
    failed = 0

    if output_root:
        if output_root.exists() and not dry_run:
            raise FileExistsError(f"Output already exists: {output_root}")
        for src_file in non_image_files:
            dst_file = output_root / src_file.relative_to(source_root)
            copy_non_image(src_file, dst_file, dry_run)
            copied += 1

    for src_file in image_files:
        src_ext = src_file.suffix.lower()
        if src_ext in JPEG_EXTENSIONS and not reencode_jpegs:
            if output_root:
                dst_file = output_root / src_file.relative_to(source_root)
                copy_non_image(src_file, dst_file, dry_run)
                copied += 1
            else:
                skipped += 1
            continue

        if output_root:
            dst_file = resolve_output_path(src_file, source_root, output_root)
        else:
            dst_file = src_file.with_suffix(".jpg")
            if dst_file.exists() and src_file.resolve() != dst_file.resolve() and not overwrite:
                print(f"  WARNING: target exists; skipping {src_file} -> {dst_file}")
                skipped += 1
                continue

        if convert_one(src_file, dst_file, quality, background, dry_run):
            converted += 1
            if in_place and not keep_originals and src_file.resolve() != dst_file.resolve():
                if not dry_run:
                    src_file.unlink()
                removed += 1
        else:
            failed += 1

    print()
    print("Summary:")
    print(f"  Converted: {converted}")
    print(f"  Skipped existing JPGs: {skipped}")
    print(f"  Copied non-converted files: {copied}")
    print(f"  Removed originals: {removed}")
    print(f"  Failed: {failed}")
    if output_root:
        print(f"  Output: {output_root}")

    return failed == 0


def parse_background(value):
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("background must be R,G,B")
    try:
        rgb = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("background must contain integers") from exc
    if any(v < 0 or v > 255 for v in rgb):
        raise argparse.ArgumentTypeError("background values must be 0-255")
    return rgb


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset images to JPG while preserving folder structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Dataset or image-folder root to scan")
    parser.add_argument("--output", default=None, help="Write a converted copy to this folder")
    parser.add_argument("--in-place", action="store_true", help="Convert files inside --source")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality, 1-100")
    parser.add_argument("--keep-originals", action="store_true", help="In-place mode: keep source PNG/TIFF/etc files")
    parser.add_argument("--reencode-jpegs", action="store_true", help="Also rewrite existing .jpg/.jpeg files")
    parser.add_argument("--overwrite", action="store_true", help="In-place mode: overwrite an existing same-stem .jpg")
    parser.add_argument("--background", type=parse_background, default=(0, 0, 0),
                        help="RGB background for transparent images, e.g. 0,0,0 or 255,255,255")
    parser.add_argument("--dry-run", action="store_true", help="Preview work without writing files")
    args = parser.parse_args()

    quality = max(1, min(100, args.quality))
    ok = convert_dataset(
        source=args.source,
        output=args.output,
        in_place=args.in_place,
        quality=quality,
        keep_originals=args.keep_originals,
        reencode_jpegs=args.reencode_jpegs,
        overwrite=args.overwrite,
        background=args.background,
        dry_run=args.dry_run,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
