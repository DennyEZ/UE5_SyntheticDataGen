"""
YOLO Detection Dataset Verification Script - V2
Overlays bounding boxes on images to verify data correctness.
Supports data.yaml formats, train/val splits, and negative samples.

Usage:
    python verify_yolo_data.py --data_path C:/UE5_YOLO_Data/

Requirements:
    pip install opencv-python pillow numpy
"""

import argparse
import os
import glob
from PIL import Image, ImageDraw, ImageFont

# Colors for different classes
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


def load_yaml_classes(data_path):
    """Parse class names from data.yaml, robust against PyYAML absence."""
    yaml_file = os.path.join(data_path, "data.yaml")
    class_names = {}
    
    if not os.path.exists(yaml_file):
        print(f"WARNING: No data.yaml found in {data_path}. Bounding boxes will show IDs instead of names.")
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
                # Assuming format "0: class_name"
                if ":" in line:
                    parts = line.split(":", 1)
                    try:
                        idx = int(parts[0].strip())
                        name = parts[1].strip()
                        # Remove quotes if present
                        if name.startswith(("'", '"')) and name.endswith(("'", '"')):
                            name = name[1:-1]
                        class_names[idx] = name
                    except ValueError:
                        pass
                else:
                    # Break if we hit something that isn't a dictionary entry
                    parsing_names = False
                    
    return class_names


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
    class_name = class_names.get(class_id, f"class_{class_id}")
    
    # Draw rectangle and a small background block for the text
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
    draw.rectangle([(x1, y1 - 20), (x1 + len(class_name)*7, y1)], fill=color)
    draw.text((x1 + 2, y1 - 18), f"{class_name}", fill=(0,0,0))
    
    return image
    
def draw_negative_watermark(image):
    """Draw 'NEGATIVE BACKGROUND SAMPLE' across the middle of the image."""
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    
    text = "NEGATIVE BACKGROUND SAMPLE"
    
    # Draw a semi-transparent black overlay bar
    draw.rectangle([(0, img_height//2 - 25), (img_width, img_height//2 + 25)], fill=(0, 0, 0, 180))
    # Draw large red warning text (estimating width)
    x1 = (img_width // 2) - (len(text) * 4)
    draw.text((x1, img_height // 2 - 10), text, fill=(255, 50, 50), font_size=20)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Verify YOLO V2 dataset (YAML + Train/Val logic)")
    parser.add_argument("--data_path", type=str, default="C:/UE5_YOLO_Data/",
                        help="Path to dataset root folder containing data.yaml")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save overlay images")
    parser.add_argument("--max_images", type=int, default=10,
                        help="Maximum number of positive images to process")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="Which dataset split to verify")
    args = parser.parse_args()
    
    data_path = args.data_path
    output_path = args.output_path or os.path.join(data_path, "verify")
    os.makedirs(output_path, exist_ok=True)
    
    images_path = os.path.join(data_path, args.split, "images")
    labels_path = os.path.join(data_path, args.split, "labels")
    
    print("=" * 60)
    print("YOLO Detection Dataset V2 Verification")
    print("=" * 60)
    print(f"Data root: {data_path}")
    print(f"Target split: {args.split}/")
    print(f"Output path: {output_path}")
    print()
    
    if not os.path.exists(images_path):
        print(f"ERROR: Images path not found at {images_path}")
        print("Did you map the correct data_path root output from the generator?")
        return
    
    # Parse data.yaml
    class_names = load_yaml_classes(data_path)
    print(f"Loaded Classes: {class_names}")
    print()
    
    # Priority sort to grab some labels
    label_files = sorted(glob.glob(os.path.join(labels_path, "*.txt")))
    print(f"Found {len(label_files)} label files in {args.split}/")
    
    if len(label_files) == 0:
        print("No labels to verify! Make sure you generate the dataset first.")
        return
        
    # Attempt to grab some negative samples and positive samples to prove both work
    samples_to_process = []
    neg_count = 0
    pos_count = 0
    
    for lf in label_files:
        if os.path.getsize(lf) == 0 and neg_count < 3:
            samples_to_process.append((lf, True))
            neg_count += 1
        elif os.path.getsize(lf) > 0 and pos_count < args.max_images:
            samples_to_process.append((lf, False))
            pos_count += 1
            
        if pos_count >= args.max_images and neg_count >= 3:
            break
            
    print(f"Verifying {pos_count} positive images and {neg_count} negative images...")
    
    processed = 0
    for label_file, is_negative in samples_to_process:
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        image_file = os.path.join(images_path, f"{base_name}.png")
        
        if not os.path.exists(image_file):
            print(f"WARNING: No image for {label_file}")
            continue
        
        image = Image.open(image_file).convert("RGB")
        
        if is_negative:
            image = draw_negative_watermark(image)
        else:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            image = draw_yolo_bbox(image, class_id, x_center, y_center, 
                                                  width, height, class_names)
                        except ValueError:
                            pass
        
        output_file = os.path.join(output_path, f"{args.split}_{base_name}_verify.png")
        image.save(output_file)
        print(f"Saved: {output_file}")
        processed += 1
    
    print()
    print("=" * 60)
    print(f"Processed {processed} total images")
    print(f"Overlay images saved to: {output_path}")
    print()
    print("Check verification dir to assure bounding boxes are tight and accurate.")


if __name__ == "__main__":
    main()
