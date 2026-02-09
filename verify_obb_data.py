"""
YOLO OBB (Oriented Bounding Box) Dataset Verification Script
Overlays oriented bounding boxes on images to verify data correctness.

Usage:
    python verify_obb_data.py --data_path D:/UE5_OBB_Data/

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

# Corner colors (to show orientation)
CORNER_COLORS = [
    (255, 255, 255),  # 0: White (first corner)
    (255, 200, 200),  # 1: Light red
    (200, 255, 200),  # 2: Light green
    (200, 200, 255),  # 3: Light blue
]


def load_classes(data_path):
    """Load class names from classes.txt."""
    classes_file = os.path.join(data_path, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []


def draw_obb(image, class_id, corners_normalized, class_names):
    """Draw oriented bounding box on image."""
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    
    # Convert normalized to pixel coordinates
    corners_px = [(int(x * img_width), int(y * img_height)) for x, y in corners_normalized]
    
    if len(corners_px) != 4:
        return image
    
    # Get color and class name
    outline_color = COLORS[class_id % len(COLORS)]
    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
    
    # Draw OBB edges
    for i in range(4):
        p1 = corners_px[i]
        p2 = corners_px[(i + 1) % 4]
        draw.line([p1, p2], fill=outline_color, width=3)
    
    # Draw corner points with different colors to show orientation
    for i, (x, y) in enumerate(corners_px):
        corner_color = CORNER_COLORS[i % len(CORNER_COLORS)]
        draw.ellipse([(x-6, y-6), (x+6, y+6)], fill=corner_color, outline=(0, 0, 0))
        draw.text((x + 8, y - 8), str(i), fill=outline_color)
    
    # Draw class label at center
    cx = sum(p[0] for p in corners_px) // 4
    cy = sum(p[1] for p in corners_px) // 4
    draw.text((cx - 20, cy - 10), class_name, fill=outline_color)
    
    # Draw orientation indicator (line from corner 0 to corner 1)
    draw.line([corners_px[0], corners_px[1]], fill=(255, 255, 255), width=1)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Verify YOLO OBB dataset")
    parser.add_argument("--data_path", type=str, default="D:/UE5_OBB_Data/",
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
    print("YOLO OBB Dataset Verification")
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
                if len(parts) >= 9:  # class_id + 4 corners (8 values)
                    class_id = int(parts[0])
                    
                    # Parse 4 corner points
                    coords = [float(v) for v in parts[1:9]]
                    corners = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                    
                    image = draw_obb(image, class_id, corners, class_names)
        
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
    print("Check the overlay images - oriented boxes should align with objects.")
    print()
    print("Corner colors indicate orientation:")
    print("  0: White (first corner)")
    print("  1: Light red")
    print("  2: Light green")
    print("  3: Light blue")


if __name__ == "__main__":
    main()
