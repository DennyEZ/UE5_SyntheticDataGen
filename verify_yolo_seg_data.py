"""
YOLO Segmentation Dataset Verification Script
Overlays segmentation polygons on images to verify data correctness.

Usage:
    python verify_yolo_seg_data.py --data_path D:/UE5_YOLO_Seg_Data/

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

# Semi-transparent fill colors
FILL_COLORS = [
    (255, 0, 0, 80),
    (0, 255, 0, 80),
    (0, 0, 255, 80),
    (255, 255, 0, 80),
    (255, 0, 255, 80),
    (0, 255, 255, 80),
    (255, 165, 0, 80),
    (128, 0, 128, 80),
]


def load_classes(data_path):
    """Load class names from classes.txt."""
    classes_file = os.path.join(data_path, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []


def draw_segmentation_polygon(image, class_id, polygon_normalized, class_names):
    """Draw segmentation polygon on image with semi-transparent fill."""
    img_width, img_height = image.size
    
    # Convert normalized to pixel coordinates
    polygon_px = [(int(x * img_width), int(y * img_height)) for x, y in polygon_normalized]
    
    if len(polygon_px) < 3:
        return image
    
    # Create overlay for semi-transparent fill
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Get colors
    outline_color = COLORS[class_id % len(COLORS)]
    fill_color = FILL_COLORS[class_id % len(FILL_COLORS)]
    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
    
    # Draw filled polygon on overlay
    overlay_draw.polygon(polygon_px, fill=fill_color, outline=outline_color)
    
    # Composite overlay onto image
    image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)
    image = image.convert('RGB')
    
    # Draw outline and vertices on final image
    draw = ImageDraw.Draw(image)
    
    # Draw polygon outline
    for i in range(len(polygon_px)):
        p1 = polygon_px[i]
        p2 = polygon_px[(i + 1) % len(polygon_px)]
        draw.line([p1, p2], fill=outline_color, width=2)
    
    # Draw vertices
    for i, (x, y) in enumerate(polygon_px):
        draw.ellipse([(x-4, y-4), (x+4, y+4)], fill=outline_color, outline=(0, 0, 0))
        draw.text((x + 6, y - 6), str(i), fill=outline_color)
    
    # Draw class label at centroid
    if polygon_px:
        cx = sum(p[0] for p in polygon_px) // len(polygon_px)
        cy = sum(p[1] for p in polygon_px) // len(polygon_px)
        draw.text((cx, cy - 20), class_name, fill=outline_color)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Verify YOLO segmentation dataset")
    parser.add_argument("--data_path", type=str, default="D:/UE5_YOLO_Seg_Data/",
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
    print("YOLO Segmentation Dataset Verification")
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
                if len(parts) >= 7:  # class_id + at least 3 points (6 values)
                    class_id = int(parts[0])
                    
                    # Parse polygon points (pairs of x, y)
                    coords = [float(v) for v in parts[1:]]
                    polygon = [(coords[i], coords[i+1]) for i in range(0, len(coords)-1, 2)]
                    
                    image = draw_segmentation_polygon(image, class_id, polygon, class_names)
        
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
    print("Check the overlay images - polygons should cover the objects.")


if __name__ == "__main__":
    main()
