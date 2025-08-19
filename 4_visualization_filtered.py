"""
This code is a program for visualizing vehicle driving data.

Main features:
- Load filtered vehicle data from JSON file
- Visualize driving behaviors such as lane change, stop, normal driving
- Display colors based on vehicle speed and acceleration
- Display lane information
"""

import json
import os
import cv2
import numpy as np
import traceback
import argparse
from collections import defaultdict

# Color definitions
VEHICLE_COLORS = {
    0: (0, 255, 0),    # Green - Class 0
    1: (255, 0, 0),    # Blue - Class 1
    2: (0, 0, 255),    # Red - Class 2
    3: (255, 255, 0),  # Cyan - Class 3
}

# Color definitions by category
CATEGORY_COLORS = {
    'lane_change': (0, 0, 255),      # Red - Lane change
    'stop': (255, 0, 255),           # Magenta - Stop
    'normal_driving': (0, 255, 0),   # Green - Normal driving
    'unknown': (128, 128, 128)       # Gray - Unknown
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vehicle Driving Data Visualization System')
    parser.add_argument('--video', '-v', type=str, default="./video_data/received_file_20240822_101524.avi",
                        help='Path to the video file (default: ./video_data/received_file_20240822_101524.avi)')
    parser.add_argument('--json', '-j', type=str, default="./2_categorized_vehicle_data/received_file_20240822_101524.json",
                        help='Path to the categorized JSON data file (default: ./2_categorized_vehicle_data/received_file_20240822_101524.json)')
    parser.add_argument('--lane-json', '-l', type=str, default='./lane_annotations.json',
                        help='Path to the lane annotations JSON file (default: ./lane_annotations.json)')
    parser.add_argument('--output-dir', '-o', type=str, default='3_visualization_filtered_result',
                        help='Output directory for visualization results (default: 3_visualization_filtered_result)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable real-time display (only save video)')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable video saving (only display)')
    
    return parser.parse_args()

# Category color function
def get_category_color(category):
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS['unknown'])

# Speed-based colors (Red: fast, Yellow: medium, Green: slow)
def get_speed_color(speed):
    if speed > 50:
        return (0, 0, 255)  # Red - Fast
    elif speed > 20:
        return (0, 255, 255)  # Yellow - Medium
    else:
        return (0, 255, 0)  # Green - Slow

# Acceleration-based colors
def get_acceleration_color(acceleration):
    if acceleration > 10:
        return (0, 0, 255)  # Red - Rapid acceleration
    elif acceleration < -10:
        return (255, 0, 255)  # Magenta - Rapid deceleration
    else:
        return (0, 255, 0)  # Green - Normal

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

# Color generation function by label
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

# Function to group filtered data by frame
def group_data_by_frame(json_data):
    frame_groups = defaultdict(list)
    for data in json_data:
        frame_num = data.get('frame', 0)
        frame_groups[frame_num].append(data)
    return frame_groups

# Function to convert bbox to pixel coordinates
def bbox_to_pixel(bbox, width, height):
    """bbox: [center_x, center_y, width, height] -> [x1, y1, x2, y2]"""
    center_x, center_y, bbox_width, bbox_height = bbox
    
    # Check if bbox is normalized or pixel values
    if center_x <= 1.0 and center_y <= 1.0:
        # If normalized coordinates, convert to pixel coordinates
        pixel_center_x = int(center_x * width)
        pixel_center_y = int(center_y * height)
        pixel_width = int(bbox_width * width)
        pixel_height = int(bbox_height * height)
    else:
        # If already pixel coordinates
        pixel_center_x = int(center_x)
        pixel_center_y = int(center_y)
        pixel_width = int(bbox_width)
        pixel_height = int(bbox_height)
    
    x1 = pixel_center_x - pixel_width // 2
    y1 = pixel_center_y - pixel_height // 2
    x2 = pixel_center_x + pixel_width // 2
    y2 = pixel_center_y + pixel_height // 2
    
    return [x1, y1, x2, y2]

def visualize_filtered_data(video_path, filtered_json_path, lane_json_path, output_dir, no_display=False, no_save=False):
    try:
        print(f"\n{'='*50}\nStarting filtered data visualization\n{'='*50}")
        print(f"Video path: {video_path}")
        print(f"JSON data path: {filtered_json_path}")
        print(f"Lane JSON path: {lane_json_path}")
        print(f"Output directory: {output_dir}")
        print(f"No display: {no_display}")
        print(f"No save: {no_save}")
        
        # Check if files exist
        if not os.path.exists(video_path):
            print(f"Error: Video file not found - {video_path}")
            return
        
        if not os.path.exists(filtered_json_path):
            print(f"Error: JSON file not found - {filtered_json_path}")
            return
        
        if not os.path.exists(lane_json_path):
            print(f"Error: Lane JSON file not found - {lane_json_path}")
            return
        
        # Load filtered JSON data
        print(f"Loading filtered JSON data: {filtered_json_path}")
        with open(filtered_json_path, 'r') as f:
            filtered_data = json.load(f)
        
        print(f"Filtered data count: {len(filtered_data)}")
        
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
        
        # Group data by frame
        frame_groups = group_data_by_frame(filtered_data)
        print(f"Total frame count: {len(frame_groups)}")
        
        # Set result save path
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video writer
        video_writer = None
        video_out_path = None
        if not no_save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_out_path = os.path.join(output_dir, os.path.basename(filtered_json_path).split('.')[0] + "_visualization.mp4")
            video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
            print(f"Video will be saved to: {video_out_path}")
        
        frame_count = 1
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_with_visualization = frame.copy()
            
            # Draw lane areas
            for lane_id, points in lanes.items():
                if len(points) >= 3:
                    lane_color = get_lane_color(lane_id)
                    pixel_points = normalize_to_pixel(points, width, height)
                    pixel_points_array = np.array(pixel_points, np.int32)
                    
                    # Fill lane area (semi-transparent)
                    overlay = frame_with_visualization.copy()
                    cv2.fillPoly(overlay, [pixel_points_array], lane_color)
                    alpha = 0.2
                    cv2.addWeighted(overlay, alpha, frame_with_visualization, 1 - alpha, 0, frame_with_visualization)
                    
                    # Draw lane boundary lines
                    cv2.polylines(frame_with_visualization, [pixel_points_array], isClosed=True, color=lane_color, thickness=1)
            
            # Draw centerlines
            for lane_id, centerline in centerlines.items():
                if len(centerline) >= 2:
                    pixel_centerline = normalize_to_pixel(centerline, width, height)
                    for i in range(len(pixel_centerline) - 1):
                        pt1 = (int(pixel_centerline[i][0]), int(pixel_centerline[i][1]))
                        pt2 = (int(pixel_centerline[i+1][0]), int(pixel_centerline[i+1][1]))
                        cv2.line(frame_with_visualization, pt1, pt2, (255, 255, 255), 1)
            
            # Draw exclusion zones
            EXCLUSION_ZONE_COLOR = (100, 100, 100)
            for zone in exclusion_zones:
                zone_points = normalize_to_pixel(zone, width, height)
                zone_points_array = np.array(zone_points, np.int32)
                zone_points_array = zone_points_array.reshape((-1, 1, 2))
                
                overlay = frame_with_visualization.copy()
                cv2.fillPoly(overlay, [zone_points_array], EXCLUSION_ZONE_COLOR)
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, frame_with_visualization, 1 - alpha, 0, frame_with_visualization)
                cv2.polylines(frame_with_visualization, [zone_points_array], isClosed=True, color=(50, 50, 50), thickness=2)
            
            # Visualize current frame's vehicle data
            if frame_count in frame_groups:
                for vehicle_data in frame_groups[frame_count]:
                    # Extract bbox information
                    bbox = vehicle_data.get('bbox', [])
                    if len(bbox) >= 4:
                        # Convert bbox to pixel coordinates
                        x1, y1, x2, y2 = bbox_to_pixel(bbox, width, height)
                        
                        # Color based on category
                        category = vehicle_data.get('category', 'unknown')
                        category_color = get_category_color(category)
                        
                        # Draw bbox (using category color)
                        cv2.rectangle(frame_with_visualization, (x1, y1), (x2, y2), category_color, 2)
                        
                        # Display object ID
                        object_id = vehicle_data.get('object_id', 'N/A')
                        cv2.putText(frame_with_visualization, f"ID:{object_id}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, category_color, 2)
                        
                        # Display speed information
                        speed = vehicle_data.get('speed', 0.0)
                        speed_color = get_speed_color(speed)
                        cv2.putText(frame_with_visualization, f"Speed:{speed:.1f}km/h", (x1, y2+15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, speed_color, 1)
                        
                        # Display acceleration information
                        acceleration = vehicle_data.get('acceleration', 0.0)
                        accel_color = get_acceleration_color(acceleration)
                        cv2.putText(frame_with_visualization, f"Accel:{acceleration:.1f}m/s^2", (x1, y2+30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, accel_color, 1)
                        
                        # Display lane information
                        lane_id = vehicle_data.get('lane_id', 'N/A')
                        cv2.putText(frame_with_visualization, f"Lane:{lane_id}", (x1, y1-25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, category_color, 1)
                        
                        # Display category information
                        category = vehicle_data.get('category', 'unknown')
                        category_color = get_category_color(category)
                        cv2.putText(frame_with_visualization, f"Category:{category}", (x1, y1-40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, category_color, 1)
                        
                        # Draw neighbor vehicle connection lines
                        neighbors_ids = vehicle_data.get('neighbors_ids', [])
                        neighbors_distances = vehicle_data.get('neighbors_distances', [])
                        
                        for i, neighbor_id in enumerate(neighbors_ids):
                            if neighbor_id is not None and neighbor_id != '':
                                # Find neighbor vehicle
                                for other_vehicle in frame_groups[frame_count]:
                                    if other_vehicle.get('object_id') == neighbor_id:
                                        other_bbox = other_vehicle.get('bbox', [])
                                        if len(other_bbox) >= 2:
                                            # Calculate neighbor vehicle's center point
                                            if other_bbox[0] <= 1.0 and other_bbox[1] <= 1.0:
                                                other_center_x = int(other_bbox[0] * width)
                                                other_center_y = int(other_bbox[1] * height)
                                            else:
                                                other_center_x = int(other_bbox[0])
                                                other_center_y = int(other_bbox[1])
                                            
                                            # Current vehicle's center point
                                            if bbox[0] <= 1.0 and bbox[1] <= 1.0:
                                                current_center_x = int(bbox[0] * width)
                                                current_center_y = int(bbox[1] * height)
                                            else:
                                                current_center_x = int(bbox[0])
                                                current_center_y = int(bbox[1])
                                            
                                            # Draw connection line
                                            line_color = (0, 255, 255)  # Yellow

                                            # Display distance
                                            if i < len(neighbors_distances) and neighbors_distances[i] < 17.0 :
                                                cv2.line(frame_with_visualization, 
                                                        (current_center_x, current_center_y),
                                                        (other_center_x, other_center_y),
                                                        line_color, 1)
                                            
                                                distance = neighbors_distances[i]
                                                mid_x = (current_center_x + other_center_x) // 2
                                                mid_y = (current_center_y + other_center_y) // 2
                                                cv2.putText(frame_with_visualization, f"{distance:.1f}", 
                                                            (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color, 1)
                                        break
            
            # Display statistics information
            if frame_count in frame_groups:
                vehicle_count = len(frame_groups[frame_count])
                cv2.putText(frame_with_visualization, f"Vehicles: {vehicle_count}, Frame: {frame_count}, FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Save video
            if video_writer is not None:
                video_writer.write(frame_with_visualization)
            
            # Real-time screen display (optional)
            if not no_display:
                cv2.imshow("Filtered Data Visualization", frame_with_visualization)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            
            frame_count += 1
            
            # Display progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not no_display:
            cv2.destroyAllWindows()
        
        print(f"\n{'='*50}\nVisualization complete\n{'='*50}")
        if video_out_path:
            print(f"Result video save location: {video_out_path}")
        print(f"Total processed frames: {frame_count}")
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        traceback.print_exc()

def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Run visualization
        visualize_filtered_data(
            video_path=args.video,
            filtered_json_path=args.json,
            lane_json_path=args.lane_json,
            output_dir=args.output_dir,
            no_display=args.no_display,
            no_save=args.no_save
        )
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 