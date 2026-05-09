"""
YOLO Segmentation Dataset Verification Script - V2
Overlays segmentation polygons on images to verify data correctness.
Supports data.yaml formats, train/val splits, negative samples, and YOLO V3 batch mode.

Usage:
    python verify_yolo_seg_data.py --data_path D:/UE5_YOLO_Seg_Data/
    python verify_yolo_seg_data.py --yolov3

Requirements:
    pip install opencv-python pillow numpy
"""

import argparse
import os
import sys
import glob
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def find_image_for_label(images_path, base_name):
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(images_path, f"{base_name}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


# Outline colors for different classes
COLORS = [
    (255, 50, 50),    # Red
    (50, 255, 50),    # Green
    (50, 150, 255),   # Blue
    (255, 255, 50),   # Yellow
    (255, 50, 255),   # Magenta
    (50, 255, 255),   # Cyan
    (255, 165, 0),    # Orange
    (128, 50, 128),   # Purple
]

# Semi-transparent fill colors (RGBA)
FILL_COLORS = [
    (255, 50, 50, 80),
    (50, 255, 50, 80),
    (50, 150, 255, 80),
    (255, 255, 50, 80),
    (255, 50, 255, 80),
    (50, 255, 255, 80),
    (255, 165, 0, 80),
    (128, 50, 128, 80),
]


def _get_scaled_metrics(image):
    """Scale overlay sizes from the current image instead of assuming 1080p."""
    short_side = max(1, min(image.size))
    line_width = max(1, short_side // 320)
    label_font_size = max(8, short_side // 40)
    label_padding_x = max(3, label_font_size // 3)
    label_padding_y = max(2, label_font_size // 4)
    label_banner_height = label_font_size + (label_padding_y * 2)
    watermark_font_size = max(12, short_side // 24)
    return {
        "line_width": line_width,
        "label_font_size": label_font_size,
        "label_padding_x": label_padding_x,
        "label_padding_y": label_padding_y,
        "label_banner_height": label_banner_height,
        "watermark_font_size": watermark_font_size,
    }


def _load_font(font_size):
    """Try a scalable TrueType font first, then fall back to Pillow's default."""
    for font_name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _get_text_size(draw, text, font):
    """Return text width/height in pixels across Pillow versions."""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except AttributeError:
        return draw.textsize(text, font=font)


def load_yaml_classes(data_path):
    """Parse class names from data.yaml, robust against PyYAML absence."""
    yaml_file = os.path.join(data_path, "data.yaml")
    class_names = {}

    if not os.path.exists(yaml_file):
        print(f"WARNING: No data.yaml found in {data_path}. Polygons will show IDs instead of names.")
        return class_names

    parsing_names = False
    with open(yaml_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("names:"):
                parsing_names = True
                continue

            if parsing_names:
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    parts = line.split(":", 1)
                    try:
                        idx = int(parts[0].strip())
                        name = parts[1].strip()
                        if name.startswith(("'", '"')) and name.endswith(("'", '"')):
                            name = name[1:-1]
                        class_names[idx] = name
                    except ValueError:
                        pass
                else:
                    parsing_names = False

    return class_names


def draw_segmentation_polygon(image, class_id, polygon_normalized, class_names):
    """Draw segmentation polygon on image with semi-transparent fill and label banner."""
    img_width, img_height = image.size
    metrics = _get_scaled_metrics(image)
    font = _load_font(metrics["label_font_size"])

    polygon_px = [(int(x * img_width), int(y * img_height)) for x, y in polygon_normalized]
    if len(polygon_px) < 3:
        return image

    outline_color = COLORS[class_id % len(COLORS)]
    fill_color = FILL_COLORS[class_id % len(FILL_COLORS)]
    class_name = class_names.get(class_id, f"class_{class_id}")

    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.polygon(polygon_px, fill=fill_color, outline=outline_color)

    image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay).convert('RGB')

    draw = ImageDraw.Draw(image)
    for i in range(len(polygon_px)):
        p1 = polygon_px[i]
        p2 = polygon_px[(i + 1) % len(polygon_px)]
        draw.line([p1, p2], fill=outline_color, width=metrics["line_width"])

    # Label banner near the top-most vertex for readability
    xs = [p[0] for p in polygon_px]
    ys = [p[1] for p in polygon_px]
    x1 = max(0, min(xs))
    y1 = max(0, min(ys))

    text_width, text_height = _get_text_size(draw, class_name, font)
    label_left = x1
    label_top = max(0, y1 - metrics["label_banner_height"])
    label_right = min(img_width, label_left + text_width + (metrics["label_padding_x"] * 2))
    label_bottom = min(img_height, label_top + metrics["label_banner_height"])
    draw.rectangle([(label_left, label_top), (label_right, label_bottom)], fill=outline_color)

    text_x = label_left + metrics["label_padding_x"]
    text_y = label_top + max(0, (metrics["label_banner_height"] - text_height) // 2 - 1)
    draw.text((text_x, text_y), class_name, fill=(0, 0, 0), font=font)

    return image


def draw_negative_watermark(image):
    """Draw 'NEGATIVE BACKGROUND SAMPLE' across the middle of the image."""
    metrics = _get_scaled_metrics(image)
    font = _load_font(metrics["watermark_font_size"])
    img_width, img_height = image.size
    text = "NEGATIVE BACKGROUND SAMPLE"

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_width, text_height = _get_text_size(draw, text, font)

    watermark_padding_y = max(4, metrics["watermark_font_size"] // 4)
    bar_height = max(text_height + watermark_padding_y * 4, metrics["watermark_font_size"] + watermark_padding_y * 4)
    bar_top = max(0, (img_height // 2) - (bar_height // 2))
    bar_bottom = min(img_height, bar_top + bar_height)
    draw.rectangle([(0, bar_top), (img_width, bar_bottom)], fill=(0, 0, 0, 180))

    text_x = max(0, (img_width - text_width) // 2)
    text_y = max(0, bar_top + ((bar_bottom - bar_top - text_height) // 2) - 1)
    draw.text((text_x, text_y), text, fill=(255, 50, 50, 255), font=font)

    composited = Image.alpha_composite(image.convert("RGBA"), overlay)
    return composited.convert("RGB")


def verify_single_dataset(
    data_path: str,
    output_path: str,
    split: str,
    max_images: int,
    prefix: str = "",
) -> int:
    """Verify one YOLO segmentation dataset folder. Returns number of images processed."""
    images_path = os.path.join(data_path, split, "images")
    labels_path = os.path.join(data_path, split, "labels")

    label = f"[{prefix}] " if prefix else ""

    print(f"{label}Data root: {data_path}")
    print(f"{label}Split: {split}/")
    print(f"{label}Output: {output_path}")

    if not os.path.exists(images_path):
        print(f"{label}SKIP — no images dir at {images_path}")
        return 0

    os.makedirs(output_path, exist_ok=True)

    class_names = load_yaml_classes(data_path)
    print(f"{label}Classes: {class_names}")

    label_files = sorted(glob.glob(os.path.join(labels_path, "*.txt")))
    print(f"{label}Found {len(label_files)} label files")

    if len(label_files) == 0:
        print(f"{label}No labels to verify.")
        return 0

    samples_to_process = []
    neg_count = 0
    pos_count = 0

    for lf in label_files:
        if os.path.getsize(lf) == 0 and neg_count < 3:
            samples_to_process.append((lf, True))
            neg_count += 1
        elif os.path.getsize(lf) > 0 and pos_count < max_images:
            samples_to_process.append((lf, False))
            pos_count += 1

        if pos_count >= max_images and neg_count >= 3:
            break

    print(f"{label}Verifying {pos_count} positive + {neg_count} negative images...")

    processed = 0
    for label_file, is_negative in samples_to_process:
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        image_file = find_image_for_label(images_path, base_name)

        if not image_file:
            print(f"{label}WARNING: No image for {label_file}")
            continue

        image = Image.open(image_file).convert("RGB")

        if is_negative:
            image = draw_negative_watermark(image)
        else:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # class_id + at least 3 points (>=7 values)
                    if len(parts) >= 7:
                        try:
                            class_id = int(parts[0])
                            coords = [float(v) for v in parts[1:]]
                            polygon = [
                                (coords[i], coords[i + 1])
                                for i in range(0, len(coords) - 1, 2)
                            ]
                            image = draw_segmentation_polygon(image, class_id, polygon, class_names)
                        except ValueError:
                            pass

        output_file = os.path.join(output_path, f"{split}_{base_name}_verify.png")
        image.save(output_file)
        print(f"{label}Saved: {output_file}")
        processed += 1

    return processed


def load_yolov3_config() -> str:
    """Import YOLO_V3_OUTPUT_ROOT from config.py in the repo root."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import importlib
    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    else:
        import config  # noqa: F401

    from config import YOLO_V3_OUTPUT_ROOT
    return YOLO_V3_OUTPUT_ROOT


def discover_yolov3_datasets(root: str) -> list:
    """Walk YOLO_V3_OUTPUT_ROOT and return (camera_group, object_name, path) tuples.

    Expected hierarchy:
        root/
        ├── cam_front_seg/
        │   ├── gate_sawfish/   ← has data.yaml or train/
        │   └── ...
        └── cam_bottom_seg/...
    """
    datasets = []
    if not os.path.isdir(root):
        return datasets

    for group in sorted(os.listdir(root)):
        group_path = os.path.join(root, group)
        if not os.path.isdir(group_path):
            continue
        for obj in sorted(os.listdir(group_path)):
            obj_path = os.path.join(group_path, obj)
            if not os.path.isdir(obj_path):
                continue
            has_split = any(
                os.path.isdir(os.path.join(obj_path, s)) for s in ("train", "val")
            )
            if has_split:
                datasets.append((group, obj, obj_path))

    return datasets


def main():
    parser = argparse.ArgumentParser(description="Verify YOLO segmentation dataset (single or V3 batch)")
    parser.add_argument("--data_path", type=str, default="D:/UE5_YOLO_Seg_Data/",
                        help="Path to dataset root folder containing data.yaml")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save overlay images")
    parser.add_argument("--max_images", type=int, default=10,
                        help="Maximum number of positive images to process per dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="Which dataset split to verify")
    parser.add_argument("--yolov3", action="store_true",
                        help="Batch-verify all objects under YOLO_V3_OUTPUT_ROOT from config.py")
    args = parser.parse_args()

    print("=" * 60)

    if args.yolov3:
        root = load_yolov3_config()
        print("YOLO V3 Segmentation Batch Verification")
        print("=" * 60)
        print(f"V3 output root: {root}")
        print()

        datasets = discover_yolov3_datasets(root)
        if not datasets:
            print(f"ERROR: No datasets found under {root}")
            print("Generate data first, or check YOLO_V3_OUTPUT_ROOT in config.py.")
            return

        print(f"Discovered {len(datasets)} object dataset(s):")
        for group, obj, _ in datasets:
            print(f"  {group}/{obj}")
        print()

        total_processed = 0
        for group, obj, obj_path in datasets:
            tag = f"{group}/{obj}"
            out = os.path.join(obj_path, "verify")
            print("-" * 60)
            processed = verify_single_dataset(
                data_path=obj_path,
                output_path=out,
                split=args.split,
                max_images=args.max_images,
                prefix=tag,
            )
            total_processed += processed
            print()

        print("=" * 60)
        print(f"Total: {total_processed} images across {len(datasets)} datasets")

    else:
        print("YOLO Segmentation Dataset Verification")
        print("=" * 60)

        data_path = args.data_path
        output_path = args.output_path or os.path.join(data_path, "verify")

        processed = verify_single_dataset(
            data_path=data_path,
            output_path=output_path,
            split=args.split,
            max_images=args.max_images,
        )

        print()
        print("=" * 60)
        print(f"Processed {processed} total images")
        if processed > 0:
            print(f"Overlay images saved to: {output_path}")
        print()
        print("Check the overlay images — polygons should tightly cover the objects.")


if __name__ == "__main__":
    main()
