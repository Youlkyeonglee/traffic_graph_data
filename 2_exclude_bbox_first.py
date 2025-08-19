"""
Vehicle Object Filtering System

This code filters vehicle object data to perform the following tasks:
- Remove objects within exclusion zones
- Visualize lane data
- Save filtered data

Main features:
1. Load lane and exclusion zone data
2. Object filtering processing
3. Save filtered data
4. Lane visualization
"""
import json
import os
import cv2
import numpy as np
import traceback
import argparse
from collections import defaultdict # Added for defaultdict

# Lane data path settings
lane_json_path = "./lane_annotations.json"

# Lane color definitions - use different colors for each label
CENTERLINE_COLOR = (255, 255, 255)  # Centerline is white
SEQUENCE_LENGTH = 20

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vehicle Object Filtering System')
    parser.add_argument('--video', '-v', type=str, default="./video_data/received_file_20240822_101524.avi",
                        help='Path to the video file')
    parser.add_argument('--json', '-j', type=str, default="./json_data/received_file_20240822_101524.json",
                        help='Path to the JSON data file')
    parser.add_argument('--lane-json', '-l', type=str, default='./lane_annotations.json',
                        help='Path to the lane annotations JSON file (default: ./lane_annotations.json)')
    parser.add_argument('--output-dir', '-o', type=str, default='1_filtered_vehicle_data',
                        help='Output directory for filtered data (default: 1_filtered_vehicle_data)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable lane visualization mode')
    
    return parser.parse_args()

# Color generation function for each label
def get_lane_color(lane_id):
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
    try:
        if "-" in lane_id:
            index = int(lane_id.split("-")[0])
        else:
            index = int(lane_id)
    except ValueError:
        index = sum(ord(c) for c in lane_id) % len(colors)
    return colors[index % len(colors)]

# Lane data loading function
def load_lane_data(json_path):
    print(f"Attempting to load lane data: {json_path}")
    try:
        with open(json_path, 'r') as f:
            lane_data = json.load(f)
            
        lanes = {}
        centerlines = {}
        exclusion_zones = []
        
        if "lanes" in lane_data:
            for lane in lane_data["lanes"]:
                if "label" in lane and "points" in lane:
                    lane_id = lane["label"]
                    points = lane["points"]
                    lanes[lane_id] = points
                    if "centerline" in lane:
                        centerlines[lane_id] = lane["centerline"]
        if "exclusion_zones" in lane_data:
            exclusion_zones = lane_data["exclusion_zones"]
            
        print(f"Lane data loading complete: {len(lanes)} lanes, {len(exclusion_zones)} exclusion zones")
        return lanes, centerlines, exclusion_zones
    except Exception as e:
        print(f"Lane data loading error: {e}")
        traceback.print_exc()
        return {}, {}, []

# Function to convert normalized coordinates to actual pixel coordinates
def normalize_to_pixel(norm_coords, width, height):
    pixel_coords = []
    for coord in norm_coords:
        x_pixel = int(coord[0] * width)
        y_pixel = int(coord[1] * height)
        pixel_coords.append([x_pixel, y_pixel])
    return pixel_coords

# Function to check if a point is inside a polygon (Ray Casting Algorithm)
def point_in_polygon(point, polygon):
    """
    Use Ray Casting Algorithm to check if a point is inside a polygon
    point: [x, y] coordinates
    polygon: [[x1, y1], [x2, y2], ...] polygon vertices
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

# Function to check if bbox is within exclusion zones
def is_bbox_in_exclusion_zones(bbox, exclusion_zones, width, height):
    """
    bbox: [center_x, center_y, width, height] format
    exclusion_zones: list of exclusion zones
    width, height: image resolution
    """
    center_x, center_y, bbox_width, bbox_height = bbox
    
    # Check if bbox is normalized or pixel values
    if center_x <= 1.0 and center_y <= 1.0:
        # If normalized coordinates, convert to pixel coordinates
        pixel_center_x = int(center_x * width)
        pixel_center_y = int(center_y * height)
    else:
        # If already pixel coordinates
        pixel_center_x = int(center_x)
        pixel_center_y = int(center_y)
    
    # Check if bbox center point is within exclusion zone
    for zone in exclusion_zones:
        if len(zone) < 3:
            continue
        
        # Convert exclusion zone to pixel coordinates
        pixel_zone = [(int(p[0] * width), int(p[1] * height)) for p in zone]
        
        # Check if bbox center point is within zone
        if point_in_polygon([pixel_center_x, pixel_center_y], pixel_zone):
            return True
    
    return False

# Function to check if bbox is within a specific lane area and return lane ID
def get_lane_id_for_bbox(bbox, lanes, width, height):
    """
    bbox: [center_x, center_y, width, height] format
    lanes: lane data dictionary
    width, height: image resolution
    """
    center_x, center_y, bbox_width, bbox_height = bbox
    
    # Check if bbox is normalized or pixel values
    if center_x <= 1.0 and center_y <= 1.0:
        # If normalized coordinates, convert to pixel coordinates
        pixel_center_x = int(center_x * width)
        pixel_center_y = int(center_y * height)
    else:
        # If already pixel coordinates
        pixel_center_x = int(center_x)
        pixel_center_y = int(center_y)
    
    # Check each lane area to see if bbox center point is included
    for lane_id, points in lanes.items():
        if len(points) < 3:
            continue
        
        # Convert lane area to pixel coordinates
        pixel_points = [(int(p[0] * width), int(p[1] * height)) for p in points]
        
        # Check if bbox center point is within lane area
        if point_in_polygon([pixel_center_x, pixel_center_y], pixel_points):
            return lane_id
    
    return None  # If not included in any lane

# Function to remove objects within exclusion zones from JSON data
def filter_excluded_objects(json_data, exclusion_zones, lanes, width, height):
    """
    Remove objects within exclusion zones from JSON data,
    change removed object IDs to None in other objects' neighbors list,
    and update neighbors information with other objects in the same frame
    """
    filtered_data = []
    removed_count = 0
    removed_object_ids = set()  # Store removed object IDs
    lane_assignment_count = 0  # Number of objects assigned lane IDs
    
    print(f"Original data count: {len(json_data)}")
    
    # Step 1: Find and remove objects within exclusion zones and assign lane IDs
    for frame_data in json_data:
        # Check if bbox is within exclusion zones
        if 'bbox' in frame_data:
            bbox = frame_data['bbox']
            if is_bbox_in_exclusion_zones(bbox, exclusion_zones, width, height):
                removed_count += 1
                removed_object_ids.add(frame_data['object_id'])  # Store removed object ID
                continue  # Exclude this object
            
            # Assign lane ID
            lane_id = get_lane_id_for_bbox(bbox, lanes, width, height)
            if lane_id:
                frame_data['lane_id'] = lane_id
                lane_assignment_count += 1
            else:
                frame_data['lane_id'] = None
        
        filtered_data.append(frame_data)
    
    print(f"Removed object count: {removed_count}")
    print(f"Objects assigned lane IDs: {lane_assignment_count}")
    print(f"Removed object IDs: {sorted(list(removed_object_ids))}")
    
    # Step 2: Remove objects that don't exist for 5 consecutive frames or more
    frame_groups = defaultdict(list)
    for frame_data in filtered_data:
        frame_num = frame_data.get('frame', 0)
        frame_groups[frame_num].append(frame_data)
    
    # Check frame continuity for each object
    object_frames = defaultdict(list)
    for frame_data in filtered_data:
        object_id = frame_data['object_id']
        frame_num = frame_data.get('frame', 0)
        object_frames[object_id].append(frame_num)
    
    # Continuity check and removal
    short_lived_objects = set()
    for object_id, frames in object_frames.items():
        frames.sort()
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(frames)):
            if frames[i] == frames[i-1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # Remove objects that exist for less than 5 consecutive frames
        if max_consecutive < SEQUENCE_LENGTH:
            short_lived_objects.add(object_id)
            removed_object_ids.add(object_id)
    
    print(f"Objects with less than {SEQUENCE_LENGTH} consecutive frames: {len(short_lived_objects)}")
    print(f"Objects with less than {SEQUENCE_LENGTH} consecutive frames IDs: {sorted(list(short_lived_objects))}")
    
    # Remove objects with less than 5 consecutive frames
    final_filtered_data = []
    for frame_data in filtered_data:
        if frame_data['object_id'] not in short_lived_objects:
            final_filtered_data.append(frame_data)
    
    filtered_data = final_filtered_data
    print(f"Data count after continuity filtering: {len(filtered_data)}")
    
    # Step 3: Change removed object IDs to None in remaining objects' neighbors list and update
    updated_count = 0
    frame_groups = {}  # Group objects by frame
    
    # Group objects by frame
    for frame_data in filtered_data:
        frame_num = frame_data.get('frame', 0)
        if frame_num not in frame_groups:
            frame_groups[frame_num] = []
        frame_groups[frame_num].append(frame_data)
    
    # Update neighbors information for each frame
    for frame_num, frame_objects in frame_groups.items():
        for frame_data in frame_objects:
            updated = False
            
            # If neighbors_ids exists
            if 'neighbors_ids' in frame_data:
                # Change removed object IDs to None
                for i, neighbor_id in enumerate(frame_data['neighbors_ids']):
                    if neighbor_id in removed_object_ids:
                        frame_data['neighbors_ids'][i] = None
                        updated = True
                        
                        # Initialize related values
                        if 'neighbors_distances' in frame_data and i < len(frame_data['neighbors_distances']):
                            frame_data['neighbors_distances'][i] = 0.0
                        if 'neighbors_speeds' in frame_data and i < len(frame_data['neighbors_speeds']):
                            frame_data['neighbors_speeds'][i] = 0.0
                        if 'neighbors_directions' in frame_data and i < len(frame_data['neighbors_directions']):
                            frame_data['neighbors_directions'][i] = [0, 0, 0]
                        if 'neighbors_accelerations' in frame_data and i < len(frame_data['neighbors_accelerations']):
                            frame_data['neighbors_accelerations'][i] = 0.0
                        if 'neighbors_world_positions' in frame_data and i < len(frame_data['neighbors_world_positions']):
                            frame_data['neighbors_world_positions'][i] = [0, 0, 0]
                        if 'neighbors_bbox' in frame_data and i < len(frame_data['neighbors_bbox']):
                            frame_data['neighbors_bbox'][i] = [0, 0, 0, 0]
                
                # Replace None positions with new neighbor information from other objects in the same frame
                current_position = frame_data.get('position', [])
                if len(current_position) >= 3:
                    current_center_x, current_center_y, current_center_z = current_position[0], current_position[1], current_position[2]
                    
                    # Calculate distance with other objects in the same frame
                    candidates = []
                    for other_obj in frame_objects:
                        if other_obj['object_id'] != frame_data['object_id']:
                            other_position = other_obj.get('position', [])
                            if len(other_position) >= 3:
                                other_center_x, other_center_y, other_center_z = other_position[0], other_position[1], other_position[2]
                                
                                # Calculate distance using position values (3D coordinates)
                                position_distance = np.sqrt((current_center_x - other_center_x)**2 + 
                                                           (current_center_y - other_center_y)**2 + 
                                                           (current_center_z - other_center_z)**2)
                                
                                # Only add candidates if position_distance is 18.0 or less
                                if position_distance <= 18.0:
                                    candidates.append({
                                        'object_id': other_obj['object_id'],
                                        'distance': position_distance,  # Distance calculated from position values
                                        'speed': other_obj.get('speed', 0.0),
                                        'direction': other_obj.get('direction', [0, 0, 0]),
                                        'acceleration': other_obj.get('acceleration', 0.0),
                                        'position': [other_center_x, other_center_y, 0],  # Extended to 3D coordinates
                                        'bbox': other_obj.get('bbox', [0, 0, 0, 0])
                                    })
                    
                    # Sort by distance
                    candidates.sort(key=lambda x: x['distance'])
                    
                    # Fill None positions with new neighbor information
                    candidate_idx = 0
                    for i, neighbor_id in enumerate(frame_data['neighbors_ids']):
                        if neighbor_id is None and candidate_idx < len(candidates):
                            candidate = candidates[candidate_idx]
                            frame_data['neighbors_ids'][i] = candidate['object_id']
                            
                            if 'neighbors_distances' in frame_data and i < len(frame_data['neighbors_distances']):
                                frame_data['neighbors_distances'][i] = candidate['distance']
                            if 'neighbors_speeds' in frame_data and i < len(frame_data['neighbors_speeds']):
                                frame_data['neighbors_speeds'][i] = candidate['speed']
                            if 'neighbors_directions' in frame_data and i < len(frame_data['neighbors_directions']):
                                frame_data['neighbors_directions'][i] = candidate['direction']
                            if 'neighbors_accelerations' in frame_data and i < len(frame_data['neighbors_accelerations']):
                                frame_data['neighbors_accelerations'][i] = candidate['acceleration']
                            if 'neighbors_world_positions' in frame_data and i < len(frame_data['neighbors_world_positions']):
                                frame_data['neighbors_world_positions'][i] = candidate['position']
                            if 'neighbors_bbox' in frame_data and i < len(frame_data['neighbors_bbox']):
                                frame_data['neighbors_bbox'][i] = candidate['bbox']
                            
                            candidate_idx += 1
                            updated = True
            
            if updated:
                updated_count += 1
    
    print(f"Objects with updated neighbors list: {updated_count}")
    print(f"Final filtered data count: {len(filtered_data)}")
    
    return filtered_data

def process_video_with_lanes(video_path, lane_json_path):
    try:
        print(f"\n{'='*50}\nStarting lane visualization video processing\n{'='*50}")
        
        # Load lane data
        lanes, centerlines, exclusion_zones = load_lane_data(lane_json_path)
        
        # Video capture settings
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video - {video_path}")
            return
        
        # Get video information
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Set result save path
        output_dir = "lane_visualization_result"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_out_path = os.path.join(output_dir, "lane_visualization.mp4")
        # video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_with_lanes = frame.copy()
            
            # Draw lane areas
            for lane_id, points in lanes.items():
                if len(points) >= 3:
                    lane_color = get_lane_color(lane_id)
                    points_array = np.array(points, np.int32)
                    
                    # Convert normalized coordinates to pixel coordinates
                    pixel_points = normalize_to_pixel(points, width, height)
                    pixel_points_array = np.array(pixel_points, np.int32)
                    
                    # Fill lane area (semi-transparent)
                    overlay = frame_with_lanes.copy()
                    cv2.fillPoly(overlay, [pixel_points_array], lane_color)
                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, frame_with_lanes, 1 - alpha, 0, frame_with_lanes)
                    
                    # Draw lane boundary lines
                    cv2.polylines(frame_with_lanes, [pixel_points_array], isClosed=True, color=lane_color, thickness=2)
                    
                    # Display lane label
                    text = lane_id
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    label_pos = (pixel_points_array[0][0] + 10, pixel_points_array[0][1] + 20)
                    
                    # Draw text background
                    cv2.rectangle(frame_with_lanes, 
                                 (label_pos[0] - 5, label_pos[1] - text_size[1] - 5),
                                 (label_pos[0] + text_size[0] + 5, label_pos[1] + 5),
                                 (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(frame_with_lanes, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw centerlines
            for lane_id, centerline in centerlines.items():
                if len(centerline) >= 2:
                    # Convert normalized coordinates to pixel coordinates
                    pixel_centerline = normalize_to_pixel(centerline, width, height)
                    
                    # Draw centerline
                    for i in range(len(pixel_centerline) - 1):
                        pt1 = (int(pixel_centerline[i][0]), int(pixel_centerline[i][1]))
                        pt2 = (int(pixel_centerline[i+1][0]), int(pixel_centerline[i+1][1]))
                        cv2.line(frame_with_lanes, pt1, pt2, CENTERLINE_COLOR, 2)
            
            # Draw exclusion zones (if any)
            EXCLUSION_ZONE_COLOR = (100, 100, 100)
            for zone in exclusion_zones:
                zone_points = normalize_to_pixel(zone, width, height)
                zone_points_array = np.array(zone_points, np.int32)
                zone_points_array = zone_points_array.reshape((-1, 1, 2))
                
                overlay = frame_with_lanes.copy()
                cv2.fillPoly(overlay, [zone_points_array], EXCLUSION_ZONE_COLOR)
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, frame_with_lanes, 1 - alpha, 0, frame_with_lanes)
                cv2.polylines(frame_with_lanes, [zone_points_array], isClosed=True, color=(50, 50, 50), thickness=2)
            
            # Display frame information
            cv2.putText(frame_with_lanes, f"Frame: {frame_count} FPS: {int(fps)}", (10, frame_with_lanes.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save video
            # video_writer.write(frame_with_lanes)
            
            # Real-time screen display (optional)
            cv2.imshow("Lane Visualization", frame_with_lanes)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            frame_count += 1
            
            # Display progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        # video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*50}\nLane visualization complete\n{'='*50}")
        # print(f"Result video save location: {video_out_path}")
        print(f"Total processed frames: {frame_count}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        traceback.print_exc()

def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        print(f"\n{'='*50}\nStarting Exclusion Zone filtering\n{'='*50}")
        print(f"Video path: {args.video}")
        print(f"JSON data path: {args.json}")
        print(f"Lane JSON path: {args.lane_json}")
        print(f"Output directory: {args.output_dir}")
        
        # Check if files exist
        if not os.path.exists(args.video):
            print(f"Error: Video file not found - {args.video}")
            return
        
        if not os.path.exists(args.json):
            print(f"Error: JSON file not found - {args.json}")
            return
        
        if not os.path.exists(args.lane_json):
            print(f"Error: Lane JSON file not found - {args.lane_json}")
            return
        
        # Load lane data (including exclusion zones)
        lanes, centerlines, exclusion_zones = load_lane_data(args.lane_json)
        
        # Get video information (for resolution check)
        cap = cv2.VideoCapture(args.video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"Video resolution: {width}x{height}")
        
        # Load JSON data
        print(f"Loading JSON data: {args.json}")
        with open(args.json, 'r') as f:
            json_data = json.load(f)
        
        # Remove objects within exclusion zones
        filtered_data = filter_excluded_objects(json_data, exclusion_zones, lanes, width, height)
        
        # Save in new folder
        os.makedirs(args.output_dir, exist_ok=True)
        
        output_path = os.path.join(args.output_dir, os.path.basename(args.json))
        
        # Save filtered data
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"\n{'='*50}\nFiltering complete\n{'='*50}")
        print(f"Filtered data save location: {output_path}")
        print(f"Original data: {len(json_data)} items")
        print(f"Filtered data: {len(filtered_data)} items")
        print(f"Removed objects: {len(json_data) - len(filtered_data)} items")
        
        # Run visualization if requested
        if args.visualize:
            process_video_with_lanes(args.video, args.lane_json)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()




