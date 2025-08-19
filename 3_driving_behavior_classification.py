"""
Vehicle Driving Behavior Classification System

This code analyzes vehicle driving data to classify the following behaviors:
- Lane change (lane_change)
- Stop (stop)
- Normal driving (normal_driving)

Main features:
1. Load filtered JSON data
2. Classify driving behavior for each vehicle
3. Analyze driving patterns and statistics
4. Save classified data
"""
import json
import os
import cv2
import numpy as np
import traceback
import argparse
from collections import defaultdict

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vehicle Driving Behavior Classification System')
    parser.add_argument('--video', '-v', type=str, default="./video_data/received_file_20240822_101524.avi",
                        help='Path to the video file (default: ./video_data/received_file_20240822_101524.avi)')
    parser.add_argument('--json', '-j', type=str, default="./1_filtered_vehicle_data/received_file_20240822_101524.json",
                        help='Path to the filtered JSON data file (default: ./1_filtered_vehicle_data/received_file_20240822_101524.json)')
    parser.add_argument('--output-dir', '-o', type=str, default='2_categorized_vehicle_data',
                        help='Output directory for categorized data (default: 2_categorized_vehicle_data)')
    parser.add_argument('--min-speed', type=float, default=2.5,
                        help='Minimum speed threshold for stop classification (default: 2.5)')
    parser.add_argument('--sequence-length', type=int, default=7,
                        help='Sequence length for lane change detection (default: 7)')
    
    return parser.parse_args()

def load_filtered_data(json_path):
    """Load filtered JSON data."""
    print(f"Attempting to load filtered data: {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Filtered data loading complete: {len(data)} objects")
        return data
    except Exception as e:
        print(f"Data loading error: {e}")
        traceback.print_exc()
        return []

def extract_lane_number(lane_id):
    """Extract lane number from lane_id."""
    if lane_id is None:
        return None
    
    if "-" in str(lane_id):
        try:
            return int(str(lane_id).split("-")[1])
        except (ValueError, IndexError):
            return None
    return None

def classify_driving_behavior(json_data, min_speed, sequence_length):
    """
    Classify driving behavior for each object:
    - lane_change: Lane change
    - stop: Stop (speed min_speed or below)
    - normal_driving: Normal driving
    """
    print(f"\n{'='*50}\nStarting driving behavior classification\n{'='*50}")
    print(f"Using parameters: min_speed={min_speed}, sequence_length={sequence_length}")
    
    # Group frame data by object
    object_frames = defaultdict(list)
    for frame_data in json_data:
        object_id = frame_data['object_id']
        object_frames[object_id].append(frame_data)
    
    # Check lane change for each object
    lane_change_objects = set()
    
    for object_id, frames in object_frames.items():
        frames.sort(key=lambda x: x.get('frame', 0))
        
        # Fill None lane_id with previous frame's lane_id
        prev_lane_id = None
        for frame_data in frames:
            current_lane_id = frame_data.get('lane_id')
            if current_lane_id is None and prev_lane_id is not None:
                frame_data['lane_id'] = prev_lane_id
            elif current_lane_id is not None:
                prev_lane_id = current_lane_id
        
        # Extract lane numbers from each frame
        lane_numbers = []
        for frame_data in frames:
            lane_id = frame_data.get('lane_id')
            lane_number = extract_lane_number(lane_id)
            lane_numbers.append(lane_number)
        
        # Check lane number changes
        for i in range(1, len(lane_numbers)):
            prev_lane = lane_numbers[i-1]
            curr_lane = lane_numbers[i]
            
            # If lane number changed and both are valid values
            if (prev_lane is not None and curr_lane is not None and 
                prev_lane != curr_lane):
                lane_change_objects.add(object_id)
                break
    
    print(f"Objects with detected lane changes: {len(lane_change_objects)}")
    print(f"Lane change object IDs: {sorted(list(lane_change_objects))}")
    
    # Assign categories to each object
    categorized_data = []
    
    for object_id, frames in object_frames.items():
        frames.sort(key=lambda x: x.get('frame', 0))
        
        # Find lane change frames
        change_frames = []
        for i, frame_data in enumerate(frames):
            if i > 0:
                prev_lane = extract_lane_number(frames[i-1].get('lane_id'))
                curr_lane = extract_lane_number(frame_data.get('lane_id'))
                
                if (prev_lane is not None and curr_lane is not None and 
                    prev_lane != curr_lane):
                    change_frames.append(frame_data.get('frame', 0))
        
        # Calculate range before and after lane change frames
        lane_change_ranges = []
        for change_frame in change_frames:
            start_frame = max(0, change_frame - sequence_length)
            end_frame = change_frame + sequence_length
            lane_change_ranges.append((start_frame, end_frame))
        
        # Assign categories to each frame
        for frame_data in frames:
            frame_num = frame_data.get('frame', 0)
            speed = frame_data.get('speed', 0.0)
            
            # Step 1: Initialize all vehicles as normal_driving
            category = "normal_driving"
            
            # Step 2: Set frames in lane change range to lane_change
            for start_frame, end_frame in lane_change_ranges:
                if start_frame <= frame_num <= end_frame:
                    category = "lane_change"
                    break
            
            # Step 3: Set frames not in lane change range with low speed to stop
            if category == "normal_driving" and speed <= min_speed:
                category = "stop"
            
            # Add category information
            frame_data['category'] = category
            categorized_data.append(frame_data)
    
    # Statistics by category
    category_counts = defaultdict(int)
    for frame_data in categorized_data:
        category = frame_data.get('category', 'unknown')
        category_counts[category] += 1
    
    print(f"\nClassification results by category:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} items ({count/len(categorized_data)*100:.1f}%)")
    
    return categorized_data

def save_categorized_data(categorized_data, output_path):
    """Save classified data."""
    try:
        # Create output directory
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        with open(output_path, 'w') as f:
            json.dump(categorized_data, f, indent=2)
        
        print(f"\nClassified data save complete: {output_path}")
        print(f"Total {len(categorized_data)} object data saved")
        
    except Exception as e:
        print(f"Data save error: {e}")
        traceback.print_exc()

def analyze_driving_patterns(categorized_data):
    """Analyze driving patterns."""
    print(f"\n{'='*50}\nDriving Pattern Analysis\n{'='*50}")
    
    # Analyze by object
    object_patterns = defaultdict(lambda: defaultdict(int))
    
    for frame_data in categorized_data:
        object_id = frame_data['object_id']
        category = frame_data.get('category', 'unknown')
        object_patterns[object_id][category] += 1
    
    # Analyze main behavior pattern for each object
    behavior_summary = defaultdict(int)
    lane_change_objects = []
    
    for object_id, patterns in object_patterns.items():
        total_frames = sum(patterns.values())
        lane_change_ratio = patterns.get('lane_change', 0) / total_frames if total_frames > 0 else 0
        stop_ratio = patterns.get('stop', 0) / total_frames if total_frames > 0 else 0
        normal_ratio = patterns.get('normal_driving', 0) / total_frames if total_frames > 0 else 0
        
        # Determine main behavior
        if lane_change_ratio > 0.1:  # More than 10% lane change
            behavior_summary['lane_change_dominant'] += 1
            lane_change_objects.append(object_id)
        elif stop_ratio > 0.3:  # More than 30% stop
            behavior_summary['stop_dominant'] += 1
        elif normal_ratio > 0.7:  # More than 70% normal driving
            behavior_summary['normal_driving_dominant'] += 1
        else:
            behavior_summary['mixed_behavior'] += 1
    
    print(f"Driving pattern analysis results:")
    for behavior, count in behavior_summary.items():
        print(f"  {behavior}: {count} objects")
    
    print(f"\nObjects with lane change as main behavior: {sorted(lane_change_objects)}")

def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        print(f"\n{'='*50}\nStarting Driving Behavior Classification System\n{'='*50}")
        print(f"Video path: {args.video}")
        print(f"JSON data path: {args.json}")
        print(f"Output directory: {args.output_dir}")
        print(f"Min speed threshold: {args.min_speed}")
        print(f"Sequence length: {args.sequence_length}")
        
        # Check if files exist
        if not os.path.exists(args.json):
            print(f"Error: JSON file not found - {args.json}")
            return
        
        # Update global parameters
        global MIN_SPEED, SEQUENCE_LENGTH
        MIN_SPEED = args.min_speed
        SEQUENCE_LENGTH = args.sequence_length
        
        # 1. Load filtered data
        json_data = load_filtered_data(args.json)
        if not json_data:
            print("Data loading failed")
            return
        
        # 2. Classify driving behavior
        categorized_data = classify_driving_behavior(json_data, args.min_speed, args.sequence_length)
        
        # 3. Analyze driving patterns
        analyze_driving_patterns(categorized_data)
        
        # 4. Save classified data
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, os.path.basename(args.json))
        save_categorized_data(categorized_data, output_path)
        
        print(f"\n{'='*50}\nDriving behavior classification complete\n{'='*50}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 