"""
YOLO Detection Dataset Verification Script
Overlays bounding boxes on images to verify data correctness.

Usage:
    python verify_yolo_data.py --data_path D:/UE5_YOLO_Data/

Requirements:
    pip install opencv-python pillow numpy
"""

import argparse
import os
import glob
from PIL import Image, ImageDraw

# Colors for different classes
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
]


def load_classes(data_path):
    """Load class names from classes.txt."""
    classes_file = os.path.join(data_path, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []


def draw_yolo_bbox(image, class_id, x_center, y_center, width, height, class_names):
    """Draw YOLO bounding box on image."""
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    
    # Convert normalized to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Calculate box corners
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    # Get color and class name
    color = COLORS[class_id % len(COLORS)]
    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
    
    # Draw rectangle
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
    
    # Draw label
    draw.text((x1, y1 - 15), f"{class_name}", fill=color)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Verify YOLO detection dataset")
    parser.add_argument("--data_path", type=str, default="D:/UE5_YOLO_Data/",
                        help="Path to dataset folder")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save overlay images")
    parser.add_argument("--max_images", type=int, default=10,
                        help="Maximum number of images to process")
    args = parser.parse_args()
    
    data_path = args.data_path
    output_path = args.output_path or os.path.join(data_path, "verify")
    os.makedirs(output_path, exist_ok=True)
    
    images_path = os.path.join(data_path, "images")
    labels_path = os.path.join(data_path, "labels")
    
    print("=" * 60)
    print("YOLO Detection Dataset Verification")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print()
    
    # Load class names
    class_names = load_classes(data_path)
    print(f"Classes: {class_names}")
    print()
    
    # Find all label files
    label_files = sorted(glob.glob(os.path.join(labels_path, "*.txt")))
    print(f"Found {len(label_files)} label files")
    
    processed = 0
    for label_file in label_files:
        if processed >= args.max_images:
            break
        
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        image_file = os.path.join(images_path, f"{base_name}.png")
        
        if not os.path.exists(image_file):
            print(f"WARNING: No image for {label_file}")
            continue
        
        # Load image
        image = Image.open(image_file).convert("RGB")
        
        # Load and parse label
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    image = draw_yolo_bbox(image, class_id, x_center, y_center, 
                                          width, height, class_names)
        
        # Save overlay
        output_file = os.path.join(output_path, f"{base_name}_verify.png")
        image.save(output_file)
        print(f"Saved: {output_file}")
        
        processed += 1
    
    print()
    print("=" * 60)
    print(f"Processed {processed} images")
    print(f"Overlay images saved to: {output_path}")
    print()
    print("Check the overlay images - bounding boxes should align with objects.")


if __name__ == "__main__":
    main()
