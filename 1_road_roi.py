import cv2
import json
import numpy as np
import os
import argparse

# Global variables: Store current polygon points and overall labeling information
current_points = []
lane_annotations = []
centerline_points = []  # Points for drawing centerline
selected_lane_index = -1  # Index of area to draw centerline
is_drawing_centerline = False  # Centerline drawing mode flag
exclusion_zones = []  # List of detection exclusion zones
is_drawing_exclusion = False  # Exclusion zone drawing mode flag

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Road ROI Labeling Tool')
    parser.add_argument('--image', '-i', type=str, default="./road_input.jpg",
                        help='Path to the input image file (default: ./road_input.jpg)')
    parser.add_argument('--json', '-j', type=str, default="lane_annotations.json",
                        help='Path to save/load lane annotations JSON file (default: lane_annotations.json)')
    parser.add_argument('--output-image', '-o', type=str, default="lane_annotations_result.jpg",
                        help='Path to save the annotated result image (default: lane_annotations_result.jpg)')
    parser.add_argument('--display-width', type=int, default=1280,
                        help='Display width for visualization (default: 1280)')
    parser.add_argument('--display-height', type=int, default=720,
                        help='Display height for visualization (default: 720)')
    
    return parser.parse_args()

# Color generation function: Returns different colors based on label index
def get_color(index):
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Dark Blue
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Red
        (128, 128, 0),  # Olive
    ]
    return colors[index % len(colors)]

# Load existing labeling file
def load_annotations(json_filename):
    global exclusion_zones
    if os.path.exists(json_filename):
        try:
            with open(json_filename, "r") as f:
                data = json.load(f)
                
                # Check file format: New format is dict with "lanes" and "exclusion_zones"
                if isinstance(data, dict) and "lanes" in data:
                    exclusion_zones = data.get("exclusion_zones", [])
                    print(f"Loaded exclusion zones: {len(exclusion_zones)}")
                    # Coordinate debugging
                    if exclusion_zones and len(exclusion_zones) > 0:
                        first_zone = exclusion_zones[0]
                        print(f"First exclusion zone: {len(first_zone)} points, first point: {first_zone[0]}")
                    return data["lanes"]
                elif isinstance(data, list):
                    if data and isinstance(data[0], dict) and "exclusion_zones" in data[0]:
                        exclusion_zones = data[0]["exclusion_zones"]
                        print(f"Loaded exclusion zones from old format: {len(exclusion_zones)}")
                    else:
                        print("Exclusion zones not found in old format.")
                    return data
        except Exception as e:
            print(f"File loading error: {e}")
            import traceback
            traceback.print_exc()
    return []

# Function to draw all areas (lanes and exclusion zones)
def draw_all_annotations(canvas, annotations, with_fill=True):
    global exclusion_zones
    result = canvas.copy()
    img_height, img_width = canvas.shape[:2]
    
    # Draw lane areas (convert normalized coordinates to display size)
    for i, ann in enumerate(annotations):
        pts = np.array(ann["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = get_color(i)
        
        if with_fill:
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts.reshape(-1, 2)], color)
            alpha = 0.3  # Transparency
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)
        label_pos = (pts[0][0][0] + 10, pts[0][0][1] + 10)
        cv2.putText(result, ann["label"], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if "centerline" in ann and ann["centerline"] and len(ann["centerline"]) > 1:
            centerline_pts = np.array(ann["centerline"], np.int32)
            for j in range(len(centerline_pts) - 1):
                pt1 = (int(centerline_pts[j][0]), int(centerline_pts[j][1]))
                pt2 = (int(centerline_pts[j+1][0]), int(centerline_pts[j+1][1]))
                cv2.line(result, pt1, pt2, (255, 255, 255), 2)
    
    # Draw exclusion zones
    for i, zone in enumerate(exclusion_zones):
        # exclusion_zones are stored in normalized coordinates, so convert
        pixel_points = []
        for point in zone:
            if isinstance(point, list) or isinstance(point, tuple):
                x, y = point
                pixel_x = int(x * img_width)
                pixel_y = int(y * img_height)
                pixel_points.append((pixel_x, pixel_y))
        if len(pixel_points) >= 3:
            pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
            if with_fill:
                overlay = result.copy()
                cv2.fillPoly(overlay, [pts.reshape(-1, 2)], (0, 0, 255))
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            cv2.polylines(result, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            center_x = int(np.mean([p[0] for p in pixel_points]))
            center_y = int(np.mean([p[1] for p in pixel_points]))
            label = f"EZone {i+1}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_pos = (center_x - text_width // 2, center_y + text_height // 2)
            cv2.putText(result, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result

# Mouse click event function
def click_event(event, x, y, flags, param):
    global current_points, img, original_img, lane_annotations, centerline_points, is_drawing_centerline, selected_lane_index, is_drawing_exclusion, exclusion_zones

    if is_drawing_exclusion:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            current_points.append((x, y))
            if len(current_points) > 1:
                cv2.line(img, current_points[-2], current_points[-1], (0, 0, 255), 2)
            if len(current_points) > 2:
                temp_img = img.copy()
                cv2.line(temp_img, current_points[0], current_points[-1], (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Image", temp_img)
            else:
                cv2.imshow("Image", img)
    elif is_drawing_centerline:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
            centerline_points.append((x, y))
            if len(centerline_points) > 1:
                cv2.line(img, centerline_points[-2], centerline_points[-1], (255, 255, 255), 2)
            cv2.imshow("Image", img)
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            current_points.append((x, y))
            if len(current_points) > 1:
                cv2.line(img, current_points[-2], current_points[-1], (0, 0, 255), 2)
            cv2.imshow("Image", img)

# Check if coordinates are normalized
def is_coord_normalized(x, y):
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

# Normalize labeling data: Convert only if coordinates are not normalized
def normalize_annotations(annotations, img_width, img_height):
    normalized = []
    for ann in annotations:
        norm_ann = {"label": ann["label"], "points": []}
        for x, y in ann["points"]:
            if is_coord_normalized(x, y):
                norm_ann["points"].append((x, y))
            else:
                norm_ann["points"].append((x/img_width, y/img_height))
        if "centerline" in ann and ann["centerline"]:
            norm_ann["centerline"] = []
            for x, y in ann["centerline"]:
                if is_coord_normalized(x, y):
                    norm_ann["centerline"].append((x, y))
                else:
                    norm_ann["centerline"].append((x/img_width, y/img_height))
        else:
            norm_ann["centerline"] = []
        normalized.append(norm_ann)
    return normalized

def is_normalized(annotations):
    if not annotations:
        return False
    for ann in annotations:
        for x, y in ann["points"]:
            if x > 1.0 or y > 1.0:
                return False
        if "centerline" in ann and ann["centerline"]:
            for x, y in ann["centerline"]:
                if x > 1.0 or y > 1.0:
                    return False
    return True

# Save labeling information to JSON file (always save in new format)
def save_annotations_json(annotations, filename, exclusion_zones):
    try:
        data = {
            "lanes": annotations,
            "exclusion_zones": exclusion_zones
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        with open(filename, "r") as f:
            saved_data = json.load(f)
            if isinstance(saved_data, dict):
                ez_count = len(saved_data.get("exclusion_zones", []))
                print(f"Check: {ez_count} exclusion zones saved in JSON file.")
            else:
                print("Warning: Exclusion zones may not have been saved.")
        print(f"JSON file saved: {filename}")
    except Exception as e:
        print(f"JSON file save error: {e}")
        import traceback
        traceback.print_exc()

# Add centerline validity check function
def is_centerline_valid(points, centerline):
    """Check if centerline is located inside or near the area"""
    if not centerline:
        return False
        
    # Calculate area boundaries
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    
    # Expand boundaries slightly (20% margin)
    width = max_x - min_x
    height = max_y - min_y
    min_x -= width * 0.2
    max_x += width * 0.2
    min_y -= height * 0.2
    max_y += height * 0.2
    
    # Check if first centerline coordinate is within expanded area
    first_point = centerline[0]
    return min_x <= first_point[0] <= max_x and min_y <= first_point[1] <= max_y

# Modified screen update function
def update_display(display_img, lane_annotations, exclusion_zones=None, with_fill=True, 
                  highlight_index=-1, centerline_mode=False):
    """Function to consistently update screen reflecting current state"""
    DISPLAY_WIDTH, DISPLAY_HEIGHT = display_img.shape[1], display_img.shape[0]
    
    # Check normalization status
    is_norm = is_normalized(lane_annotations)
    
    # Prepare display annotations
    display_annotations = []
    for i, ann in enumerate(lane_annotations):
        display_ann = {
            "label": ann["label"],
            "points": []
        }
        
        # Coordinate processing
        if is_norm:
            # Convert normalized coordinates to display size
            display_ann["points"] = [(int(x * DISPLAY_WIDTH), int(y * DISPLAY_HEIGHT)) 
                                    for x, y in ann["points"]]
        else:
            # Scale pixel coordinates to fit display size
            img_height, img_width = original_img.shape[:2]
            scale_x = DISPLAY_WIDTH / img_width
            scale_y = DISPLAY_HEIGHT / img_height
            display_ann["points"] = [(int(x * scale_x), int(y * scale_y)) 
                                    for x, y in ann["points"]]
        
        # Centerline processing - add validity check
        if "centerline" in ann and ann["centerline"] and is_centerline_valid(ann["points"], ann["centerline"]):
            if is_norm:
                display_ann["centerline"] = [(int(x * DISPLAY_WIDTH), int(y * DISPLAY_HEIGHT)) 
                                          for x, y in ann["centerline"]]
            else:
                display_ann["centerline"] = [(int(x * scale_x), int(y * scale_y)) 
                                          for x, y in ann["centerline"]]
        else:
            # Don't display invalid centerlines
            if "centerline" in ann and ann["centerline"]:
                print(f"Warning: Centerline for '{ann['label']}' area is outside the area and will not be displayed.")
        
        display_annotations.append(display_ann)
    
    # Copy image and draw annotations
    img_copy = display_img.copy()
    result = draw_all_annotations(img_copy, display_annotations, with_fill=with_fill)
    
    # Highlight specific area (if selected)
    if highlight_index >= 0 and highlight_index < len(display_annotations):
        selected_pts = np.array(display_annotations[highlight_index]["points"], np.int32)
        selected_pts = selected_pts.reshape((-1, 1, 2))
        cv2.polylines(result, [selected_pts], isClosed=True, color=(0, 0, 255), thickness=3)
        
        # Highlight label of selected area
        label_pos = (selected_pts[0][0][0] + 10, selected_pts[0][0][1] + 30)
        mode_text = "Drawing centerline: " if centerline_mode else "Selected: "
        cv2.putText(result, f"{mode_text}{lane_annotations[highlight_index]['label']}", 
                    label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result

def main():
    global img, current_points, lane_annotations, original_img, centerline_points, is_drawing_centerline, selected_lane_index, exclusion_zones, is_drawing_exclusion

    # Parse command line arguments
    args = parse_arguments()
    
    # Set visualization resolution from arguments
    DISPLAY_WIDTH = args.display_width
    DISPLAY_HEIGHT = args.display_height

    # Load existing labeling
    lane_annotations = load_annotations(args.json)
    
    # Image file path from arguments
    image_path = args.image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Cannot load image: {image_path}")
        print("Please check if the file exists and the path is correct.")
        return
        
    img_height, img_width = original_img.shape[:2]
    print(f"Original image size: {img_width}x{img_height}")
    print(f"Input image: {image_path}")
    print(f"JSON file: {args.json}")
    print(f"Output image: {args.output_image}")
    print(f"Display resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    
    # Normalize loaded labeling data if not already normalized
    if lane_annotations:
        if not is_normalized(lane_annotations):
            lane_annotations = normalize_annotations(lane_annotations, img_width, img_height)
            print("Normalized loaded labeling data.")
        print(f"Loaded {len(lane_annotations)} existing labels.")
        print(f"Data normalization status: {'Normalized (0-1)' if is_normalized(lane_annotations) else 'Not normalized'}")
    
    # Resize image for display
    display_img = cv2.resize(original_img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    print(f"Visualization resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    
    # Initial screen update
    if lane_annotations:
        img = update_display(display_img, lane_annotations)
    else:
        img = display_img.copy()
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event)
    
    print(">> Specify lane areas.")
    print("   - Mouse left click: Add polygon points")
    print("   - 'n' key: Save current area and specify next area")
    print("   - 'q' key: Exit and save JSON file")
    print("   - 'd' key: Delete last added area")
    print("   - 'r' key: Delete area by label")
    print("   - 'l' key: Draw centerline for selected area")
    print("   - 'm' key: Normalize all coordinates to 0-1 range")
    print("   - 't' key: Toggle exclusion zone drawing mode")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        # 'm' key: Normalization processing
        if key == ord("m"):
            if not lane_annotations:
                print("No data to normalize.")
                continue
            if is_normalized(lane_annotations):
                print("All data is already normalized (0-1 range).")
                continue
            lane_annotations = normalize_annotations(lane_annotations, img_width, img_height)
            print("Non-normalized coordinates have been normalized to 0-1 range.")
            save_annotations_json(lane_annotations, args.json, exclusion_zones)
            img = update_display(display_img, lane_annotations)
            cv2.imshow("Image", img)
        
        # 'n' key: Save new area (normalize display coordinates before saving)
        elif key == ord("n"):
            if len(current_points) < 3:
                print("At least 3 points are needed to create a polygon.")
                continue
            lane_label = input("Enter label for this area: ")
            normalized_points = [(x / float(DISPLAY_WIDTH), y / float(DISPLAY_HEIGHT)) for x, y in current_points]
            lane_annotations.append({
                "label": lane_label,
                "points": normalized_points,
                "centerline": []
            })
            print(f"Area with label '{lane_label}' has been saved.")
            current_points = []
            img = update_display(display_img, lane_annotations)
            cv2.imshow("Image", img)
        
        # 'd' key: Delete last area
        elif key == ord("d"):
            if lane_annotations:
                removed = lane_annotations.pop()
                print(f"Area '{removed['label']}' has been deleted.")
                img = update_display(display_img, lane_annotations)
                cv2.imshow("Image", img)
                print(f"Remaining areas: {len(lane_annotations)}")
            else:
                print("No area to delete.")
        
        # 'r' key: Delete area by label
        elif key == ord("r"):
            if not lane_annotations:
                print("No area to delete.")
                continue
            existing_labels = [f"{i}: {ann['label']}" for i, ann in enumerate(lane_annotations)]
            print("Current area list:")
            for label in existing_labels:
                print(f"  {label}")
            label_to_delete = input("Enter label of area to delete: ")
            deleted_count = 0
            original_count = len(lane_annotations)
            for i in range(len(lane_annotations)-1, -1, -1):
                if lane_annotations[i]["label"] == label_to_delete:
                    del lane_annotations[i]
                    deleted_count += 1
            if deleted_count > 0:
                print(f"{deleted_count} area(s) with label '{label_to_delete}' have been deleted.")
                img = update_display(display_img, lane_annotations)
                cv2.imshow("Image", img)
                print(f"Areas {original_count} → {len(lane_annotations)}")
            else:
                print(f"No area with label '{label_to_delete}' found.")
        
        # 'l' key: Switch to centerline drawing mode
        elif key == ord("l"):
            if not lane_annotations:
                print("No area to draw centerline. Create an area first.")
                continue
            existing_labels = [f"{i}: {ann['label']}" for i, ann in enumerate(lane_annotations)]
            print("Current area list:")
            for label in existing_labels:
                print(f"  {label}")
            try:
                selected_lane_index = int(input("Enter number of area to draw centerline: "))
                if selected_lane_index < 0 or selected_lane_index >= len(lane_annotations):
                    print("Invalid area number.")
                    continue
            except ValueError:
                print("Please enter a number.")
                continue
            is_drawing_centerline = True
            centerline_points = []
            img = update_display(display_img, lane_annotations, highlight_index=selected_lane_index, centerline_mode=True)
            cv2.imshow("Image", img)
            print(f"Drawing centerline for '{lane_annotations[selected_lane_index]['label']}' area.")
            print("  - Mouse left click: Add centerline points")
            print("  - 'c' key: Complete centerline drawing")
            print("  - 'x' key: Cancel centerline drawing")
        
        # Press 'c' key to save centerline
        elif key == ord("c") and is_drawing_centerline:
            if len(centerline_points) < 2:
                print("Centerline needs at least 2 points.")
                continue
            
            # Validate centerline before saving
            normalized_centerline = [(x/DISPLAY_WIDTH, y/DISPLAY_HEIGHT) for x, y in centerline_points]
            if is_centerline_valid(lane_annotations[selected_lane_index]["points"], normalized_centerline):
                # Save centerline
                lane_annotations[selected_lane_index]["centerline"] = normalized_centerline
                print(f"Centerline has been saved for '{lane_annotations[selected_lane_index]['label']}' area.")
                centerline_points = []
                
                # Check if centerlines have been drawn for all areas
                has_all_centerlines = True
                for ann in lane_annotations:
                    if "centerline" not in ann or not ann["centerline"]:
                        has_all_centerlines = False
                        break
                
                if has_all_centerlines:
                    print("Centerlines have been drawn for all areas!")
            else:
                print("Warning: Centerline is outside the area. Please draw centerline inside the area.")
            continue
        
        # 'x' key: Cancel centerline
        elif key == ord("x") and is_drawing_centerline:
            print("Centerline drawing has been cancelled.")
            is_drawing_centerline = False
            selected_lane_index = -1
            centerline_points = []
            img = update_display(display_img, lane_annotations)
            cv2.imshow("Image", img)
        
        # 't' key: Toggle exclusion zone drawing mode
        elif key == ord('t'):
            if is_drawing_exclusion:
                if current_points and len(current_points) >= 3:
                    normalized_points = [(x / float(DISPLAY_WIDTH), y / float(DISPLAY_HEIGHT)) for x, y in current_points]
                    exclusion_zones.append(normalized_points)
                    print(f"Exclusion zone added. (Total {len(exclusion_zones)})")
                    print(f"Exclusion zone points: {len(normalized_points)}")
                current_points.clear()
                is_drawing_exclusion = False
            else:
                is_drawing_exclusion = True
                is_drawing_centerline = False
                current_points.clear()
                print("Starting exclusion zone drawing mode. Click polygon points then press 't' again to complete.")
            img = update_display(display_img, lane_annotations)
            cv2.imshow("Image", img)
        
        # 'q' key: Exit
        elif key == ord("q"):
            break

    # Save final labeling image
    if lane_annotations:
        result_img = draw_all_annotations(original_img, 
                        [ { "label": ann["label"],
                            "points": [(int(x * img_width), int(y * img_height)) for x, y in ann["points"]],
                            "centerline": [(int(x * img_width), int(y * img_height)) for x, y in ann["centerline"]]
                          } for ann in lane_annotations ], with_fill=True)
        cv2.imwrite(args.output_image, result_img)
        print(f"Labeling image has been saved to '{args.output_image}'.")
        normalized_status = "Normalized (0-1)" if is_normalized(lane_annotations) else "Not normalized (pixel coordinates)"
        save_annotations_json(lane_annotations, args.json, exclusion_zones)
        print(f"All labeling information has been saved to '{args.json}'. ({normalized_status})")
        print(f"Total {len(lane_annotations)} areas have been saved.")
    else:
        print("No labeling to save.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
