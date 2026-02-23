"""
YOLO Segmentation Model Verification Script
Runs inference on images/videos and saves annotated results with predicted masks.

Use this to visually verify your trained model's accuracy on test data.

Usage:
    python modeltest/verify_model.py --model best.pt --input test_images/ --output results/
    python modeltest/verify_model.py --model best.pt --input test_videos/ --conf 0.5

Requirements:
    pip install ultralytics opencv-python
"""

import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

# Default paths (relative to this script's directory)
# USAGE: python modeltest/verify_model.py
#   or:  python modeltest/verify_model.py --model my_model.pt --conf 0.5
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = str(SCRIPT_DIR / "best.pt")
DEFAULT_INPUT = str(SCRIPT_DIR / "test_sources")
DEFAULT_OUTPUT = str(SCRIPT_DIR / "test_results")

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics is required. Install via: pip install ultralytics")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: opencv-python is required. Install via: pip install opencv-python")
    sys.exit(1)

# Supported file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Segmentation Model Verification — "
                    "Run inference and overlay predicted masks on images/videos"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Path to trained YOLO model (default: {DEFAULT_MODEL})")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"Folder containing input images/videos (default: test_sources/)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output folder for annotated results (default: test_results/)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold for NMS (default: 0.7)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--save-txt", action="store_true",
                        help="Also save predicted labels as .txt files")
    parser.add_argument("--no-conf", action="store_true",
                        help="Hide confidence scores on overlay")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide class labels on overlay")
    parser.add_argument("--line-width", type=int, default=2,
                        help="Mask outline thickness (default: 2)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to process (default: all)")
    return parser.parse_args()


def collect_files(input_dir):
    """Scan input directory for images and videos."""
    images = []
    videos = []
    for f in sorted(Path(input_dir).rglob("*")):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(f)
            elif ext in VIDEO_EXTENSIONS:
                videos.append(f)
    return images, videos


def save_predictions_txt(result, output_path):
    """Save predicted masks as YOLO-format .txt label file."""
    txt_path = output_path.with_suffix(".txt")
    lines = []

    if result.masks is not None and result.boxes is not None:
        masks_xy = result.masks.xyn  # normalized polygon coordinates
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for cls_id, conf, polygon in zip(classes, confs, masks_xy):
            if len(polygon) < 3:
                continue
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
            lines.append(f"{cls_id} {coords}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    return len(lines)


def process_images(model, images, output_dir, args):
    """Run inference on images and save annotated results."""
    stats = defaultdict(lambda: {"count": 0, "total_conf": 0.0})
    total_predictions = 0
    processed = 0

    labels_dir = output_dir / "labels" if args.save_txt else None
    if labels_dir:
        labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        if args.max_images and processed >= args.max_images:
            break

        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.img_size,
            verbose=False,
        )

        result = results[0]

        # Draw annotated frame using ultralytics built-in renderer
        annotated = result.plot(
            conf=not args.no_conf,
            labels=not args.no_labels,
            line_width=args.line_width,
        )

        # Save annotated image
        out_path = output_dir / f"{img_path.stem}_pred{img_path.suffix}"
        cv2.imwrite(str(out_path), annotated)

        # Save predictions as txt
        if args.save_txt:
            save_predictions_txt(result, labels_dir / img_path.stem)

        # Collect stats
        num_preds = 0
        if result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            num_preds = len(classes)
            for cls_id, conf in zip(classes, confs):
                name = model.names.get(cls_id, f"class_{cls_id}")
                stats[name]["count"] += 1
                stats[name]["total_conf"] += conf

        total_predictions += num_preds
        processed += 1

        print(f"  [{processed}/{len(images)}] {img_path.name} → "
              f"{num_preds} detection(s)")

    return processed, total_predictions, stats


def process_videos(model, videos, output_dir, args):
    """Run inference on videos frame-by-frame and save annotated videos."""
    stats = defaultdict(lambda: {"count": 0, "total_conf": 0.0})
    total_predictions = 0
    total_frames = 0

    for vid_path in videos:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"  WARNING: Cannot open video: {vid_path.name}")
            continue

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video writer
        out_path = output_dir / f"{vid_path.stem}_pred.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        print(f"  Processing video: {vid_path.name} ({frame_count} frames, "
              f"{width}x{height} @ {fps:.1f} fps)")

        frame_idx = 0
        vid_preds = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on frame
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.img_size,
                verbose=False,
            )

            result = results[0]

            # Draw annotated frame
            annotated = result.plot(
                conf=not args.no_conf,
                labels=not args.no_labels,
                line_width=args.line_width,
            )

            writer.write(annotated)

            # Collect stats
            if result.boxes is not None:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                num_preds = len(classes)
                vid_preds += num_preds
                for cls_id, conf in zip(classes, confs):
                    name = model.names.get(cls_id, f"class_{cls_id}")
                    stats[name]["count"] += 1
                    stats[name]["total_conf"] += conf

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                print(f"    Frame {frame_idx}/{frame_count} "
                      f"({fps_actual:.1f} fps)")

        cap.release()
        writer.release()

        elapsed = time.time() - start_time
        total_predictions += vid_preds
        total_frames += frame_idx

        print(f"  → Saved: {out_path.name} "
              f"({frame_idx} frames, {vid_preds} detections, "
              f"{elapsed:.1f}s)")

    return total_frames, total_predictions, stats


def print_summary(num_images, num_videos, total_frames, total_predictions,
                  stats, elapsed):
    """Print summary statistics."""
    print()
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Images processed:      {num_images}")
    print(f"  Videos processed:      {num_videos}")
    if num_videos > 0:
        print(f"  Total video frames:    {total_frames}")
    print(f"  Total predictions:     {total_predictions}")
    print(f"  Time elapsed:          {elapsed:.1f}s")
    print()

    if stats:
        print("  Per-class breakdown:")
        print(f"  {'Class':<25} {'Count':>8} {'Avg Conf':>10}")
        print(f"  {'-'*25} {'-'*8} {'-'*10}")
        for name in sorted(stats.keys()):
            s = stats[name]
            avg_conf = s["total_conf"] / s["count"] if s["count"] > 0 else 0
            print(f"  {name:<25} {s['count']:>8} {avg_conf:>10.3f}")
    else:
        print("  No predictions made — model may need more training "
              "or confidence threshold may be too high.")

    print()
    print("=" * 60)


def main():
    args = parse_args()

    # Validate inputs
    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("YOLO SEGMENTATION MODEL VERIFICATION")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Input:       {args.input}")
    print(f"  Output:      {output_dir}")
    print(f"  Confidence:  {args.conf}")
    print(f"  IoU:         {args.iou}")
    print(f"  Image size:  {args.img_size}")
    print()

    # Load model
    print("Loading model...")
    model = YOLO(args.model)
    print(f"  Model loaded: {len(model.names)} classes")
    print(f"  Classes: {model.names}")
    print()

    # Collect input files
    images, videos = collect_files(args.input)
    print(f"Found {len(images)} image(s) and {len(videos)} video(s)")
    print()

    start_time = time.time()
    total_predictions = 0
    all_stats = defaultdict(lambda: {"count": 0, "total_conf": 0.0})
    total_frames = 0
    num_images = 0
    num_videos = 0

    # Process images
    if images:
        print("Processing images...")
        num_images, img_preds, img_stats = process_images(
            model, images, output_dir, args
        )
        total_predictions += img_preds
        for name, s in img_stats.items():
            all_stats[name]["count"] += s["count"]
            all_stats[name]["total_conf"] += s["total_conf"]

    # Process videos
    if videos:
        print("Processing videos...")
        total_frames, vid_preds, vid_stats = process_videos(
            model, videos, output_dir, args
        )
        num_videos = len(videos)
        total_predictions += vid_preds
        for name, s in vid_stats.items():
            all_stats[name]["count"] += s["count"]
            all_stats[name]["total_conf"] += s["total_conf"]

    elapsed = time.time() - start_time

    # Print summary
    print_summary(num_images, num_videos, total_frames, total_predictions,
                  all_stats, elapsed)

    print(f"Annotated outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
