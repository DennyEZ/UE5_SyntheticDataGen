"""
DOPE Dataset Verification Script
Overlays projected cuboid points on images to verify data correctness.

Usage:
    python verify_dope_data.py --data_path D:/UE5_Data/

Requirements:
    pip install opencv-python pillow numpy
"""

import argparse
import json
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import sys

# Cuboid corner colors (matching DOPE's convention)
COLORS = [
    (255, 255, 0),    # 0: Yellow
    (255, 0, 255),    # 1: Magenta
    (0, 0, 255),      # 2: Blue
    (255, 0, 0),      # 3: Red
    (0, 255, 0),      # 4: Green
    (255, 165, 0),    # 5: Orange
    (139, 69, 19),    # 6: Brown
    (0, 255, 255),    # 7: Cyan
    (255, 255, 255),  # 8: White (centroid)
]

# Cuboid edges (pairs of corner indices to connect)
EDGES = [
    (0, 1), (1, 5), (5, 4), (4, 0),  # Front face
    (2, 3), (3, 7), (7, 6), (6, 2),  # Back face
    (0, 2), (1, 3), (4, 6), (5, 7),  # Connecting edges
]


def draw_cuboid(image, projected_cuboid, obj_class):
    """Draw cuboid overlay on image."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Draw edges
    for i, j in EDGES:
        if i < len(projected_cuboid) and j < len(projected_cuboid):
            p1 = projected_cuboid[i]
            p2 = projected_cuboid[j]
            
            # Check if points are valid (not behind camera)
            if p1[0] > -9000 and p2[0] > -9000:
                # Clamp to image bounds for drawing
                x1 = max(0, min(width-1, int(p1[0])))
                y1 = max(0, min(height-1, int(p1[1])))
                x2 = max(0, min(width-1, int(p2[0])))
                y2 = max(0, min(height-1, int(p2[1])))
                draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=2)
    
    # Draw corner points
    RADIUS = 5
    for idx, pt in enumerate(projected_cuboid):
        if pt[0] > -9000:  # Valid point
            x = int(pt[0])
            y = int(pt[1])
            
            # Skip if outside image
            if 0 <= x < width and 0 <= y < height:
                color = COLORS[idx] if idx < len(COLORS) else (255, 255, 255)
                draw.ellipse(
                    [(x - RADIUS, y - RADIUS), (x + RADIUS, y + RADIUS)],
                    fill=color,
                    outline=(0, 0, 0)
                )
                # Draw index number
                draw.text((x + RADIUS + 2, y - RADIUS), str(idx), fill=color)
    
    # Draw class label
    if projected_cuboid and projected_cuboid[8][0] > -9000:
        centroid = projected_cuboid[8]
        x, y = int(centroid[0]), int(centroid[1])
        if 0 <= x < width and 0 <= y < height:
            draw.text((x + 10, y - 20), obj_class, fill=(255, 255, 0))
    
    return image


def validate_json_structure(json_data):
    """Validate JSON matches DOPE format."""
    issues = []
    
    # Check camera_data
    if "camera_data" not in json_data:
        issues.append("Missing 'camera_data'")
    else:
        cam = json_data["camera_data"]
        for field in ["width", "height", "intrinsics"]:
            if field not in cam:
                issues.append(f"Missing camera_data.{field}")
        
        if "intrinsics" in cam:
            for field in ["fx", "fy", "cx", "cy"]:
                if field not in cam["intrinsics"]:
                    issues.append(f"Missing camera_data.intrinsics.{field}")
    
    # Check objects
    if "objects" not in json_data:
        issues.append("Missing 'objects'")
    else:
        for i, obj in enumerate(json_data["objects"]):
            for field in ["class", "visibility", "location", "quaternion_xyzw", "projected_cuboid"]:
                if field not in obj:
                    issues.append(f"Object {i}: Missing '{field}'")
            
            if "projected_cuboid" in obj:
                if len(obj["projected_cuboid"]) != 9:
                    issues.append(f"Object {i}: projected_cuboid should have 9 points, has {len(obj['projected_cuboid'])}")
            
            if "location" in obj:
                if len(obj["location"]) != 3:
                    issues.append(f"Object {i}: location should have 3 values")
            
            if "quaternion_xyzw" in obj:
                if len(obj["quaternion_xyzw"]) != 4:
                    issues.append(f"Object {i}: quaternion_xyzw should have 4 values")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="Verify DOPE dataset")
    parser.add_argument("--data_path", type=str, default="D:/UE5_Data/",
                        help="Path to dataset folder")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save overlay images (default: data_path/verify/)")
    parser.add_argument("--max_images", type=int, default=10,
                        help="Maximum number of images to process")
    args = parser.parse_args()
    
    data_path = args.data_path
    output_path = args.output_path or os.path.join(data_path, "verify")
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 60)
    print("DOPE Dataset Verification")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print()
    
    # Find all JSON files
    json_files = sorted(glob.glob(os.path.join(data_path, "*.json")))
    print(f"Found {len(json_files)} JSON files")
    
    # Find all image files
    png_files = sorted(glob.glob(os.path.join(data_path, "*.png")))
    print(f"Found {len(png_files)} PNG files")
    print()
    
    # Validate each pair
    valid_count = 0
    issues_count = 0
    
    processed = 0
    for json_file in json_files:
        if processed >= args.max_images:
            break
        
        # Find corresponding image
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        image_file = os.path.join(data_path, f"{base_name}.png")
        
        if not os.path.exists(image_file):
            print(f"WARNING: No image for {json_file}")
            issues_count += 1
            continue
        
        # Load and validate JSON
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        issues = validate_json_structure(json_data)
        if issues:
            print(f"ISSUES in {base_name}.json:")
            for issue in issues:
                print(f"  - {issue}")
            issues_count += 1
        else:
            valid_count += 1
        
        # Create overlay image
        image = Image.open(image_file).convert("RGB")
        
        for obj in json_data.get("objects", []):
            cuboid = obj.get("projected_cuboid", [])
            obj_class = obj.get("class", "unknown")
            image = draw_cuboid(image, cuboid, obj_class)
        
        # Save overlay
        output_file = os.path.join(output_path, f"{base_name}_verify.png")
        image.save(output_file)
        print(f"Saved: {output_file}")
        
        processed += 1
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Valid JSON files: {valid_count}")
    print(f"Files with issues: {issues_count}")
    print(f"Overlay images saved to: {output_path}")
    print()
    print("Check the overlay images - the colored dots should align with")
    print("the corners of your objects in the images.")
    print()
    print("Corner index colors:")
    for i, color in enumerate(COLORS):
        print(f"  {i}: RGB{color}")


if __name__ == "__main__":
    main()
