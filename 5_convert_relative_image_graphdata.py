import os
import json
import pickle
import torch
from torch_geometric.data import Data
import sys
import time  # Processing time measurement
import psutil  # Memory usage monitoring (requires pip install psutil)
from tqdm import tqdm  # Progress display (requires pip install tqdm)

# Memory usage check function
def get_memory_usage():
    """Returns the current process memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Return in MB

def extract_node_features(json_data):
    node_features = []
    node_id_map = {}
    node_index = 0
    for entry in json_data:
        obj_id = entry["object_id"]
        frame = entry["frame"]
        key = (obj_id, frame)
        if key not in node_id_map:
            node_id_map[key] = node_index
            node_index += 1
        
        # Extract fields according to new JSON format
        bbox = entry["bbox"]  # [center_x, center_y, width, height]
        speed = [entry["speed"]]
        direction = entry["direction"]  # [dx, dy, dz]
        acceleration = [entry["acceleration"]]
        
        # Combine all features (including additional information from existing data)
        features = bbox + speed + direction + acceleration # 9-dimensional
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    return x, node_id_map

def extract_labels(json_data):
    label_map = {
        "stop": 0,
        "lane_change": 1,
        "normal_driving": 2
    }
    
    # Count categories before filtering
    all_categories = {}
    for entry in json_data:
        cat = entry["category"]
        all_categories[cat] = all_categories.get(cat, 0) + 1
    
    print(f"Category distribution (before filtering):")
    for cat, count in all_categories.items():
        print(f"  {cat}: {count}")
    # Filter only entries with the categories in the mapping
    filtered_data = [entry for entry in json_data if entry["category"] in label_map]
    
    print(f"Original data had {len(json_data)} entries")
    print(f"Filtered data has {len(filtered_data)} entries")
    print(f"Removed {len(json_data) - len(filtered_data)} entries with unsupported categories")
    
    # Extract labels only from valid entries
    labels = [label_map[entry["category"]] for entry in filtered_data]
    y = torch.tensor(labels, dtype=torch.long)
    return y, filtered_data  # Return filtered data for further processing

def extract_edge_index(json_data, node_id_map, max_neighbors=4):
    """
    Construct edges using neighbors_ids for up to max_neighbors neighboring vehicles per vehicle
    Sort by neighbors_distances in ascending order and select
    
    Parameters:
    - max_neighbors: Maximum number of neighboring vehicles per vehicle (default: 4)
    """
    edge_list = []
    missing_neighbors = 0
    
    for entry in json_data:
        src_id = entry["object_id"]
        src_frame = entry["frame"]
        src_key = (src_id, src_frame)
        
        if src_key not in node_id_map:
            continue
            
        src_idx = node_id_map[src_key]
        
        # Process neighbors_ids and neighbors_distances together
        if "neighbors_ids" in entry and "neighbors_distances" in entry:
            neighbors_ids = entry["neighbors_ids"]
            neighbors_distances = entry["neighbors_distances"]
            
            # Create valid neighbors as (distance, index, ID) tuples
            valid_neighbors = []
            for i, (nbr_id, distance) in enumerate(zip(neighbors_ids, neighbors_distances)):
                # Skip empty strings or None values
                if not nbr_id or nbr_id == "" or distance is None:
                    continue
                    
                # Convert string ID to integer
                try:
                    nbr_id = int(nbr_id)
                    nbr_key = (nbr_id, src_frame)
                    
                    if nbr_key in node_id_map:
                        valid_neighbors.append((distance, i, nbr_id))
                    else:
                        missing_neighbors += 1
                except (ValueError, TypeError):
                    continue
            
            # Sort by distance (ascending order)
            valid_neighbors.sort(key=lambda x: x[0])
            # print(valid_neighbors)
            
            # Select only max_neighbors
            for distance, original_idx, nbr_id in valid_neighbors[:max_neighbors]:
                nbr_key = (nbr_id, src_frame)
                dst_idx = node_id_map[nbr_key]
                edge_list.append([src_idx, dst_idx])
            # print(edge_list)
            # sys.exit()
    # Check if edge list is empty
    if not edge_list:
        print("Warning: No edges found in data.")
        return torch.zeros((2, 0), dtype=torch.long)
    
    # Convert edge list to tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(f"Total {edge_index.size(1)} edges generated. (Max neighbors: {max_neighbors})")
    if missing_neighbors > 0:
        print(f"Note: {missing_neighbors} neighbor IDs could not be found in node map.")
    return edge_index

def extract_edge_attr(json_data, node_id_map, max_neighbors=4, max_edge=5):
    """
    Extract relationship information with each neighboring vehicle as edge attributes (using relative values)
    Sort by neighbors_distances in ascending order and select
    Attributes: relative bbox(4), relative speed(1), relative acceleration(1) - Total 6-dimensional
    
    Parameters:
    - max_neighbors: Maximum number of neighboring vehicles per vehicle (default: 4)
    """
    print("\n🔗 Edge attribute extraction starting...")
    start_time = time.time()

    edge_attrs = []
    edge_list = []
    
    # Create dictionary for fast lookup
    node_lookup = {}
    for entry in json_data:
        key = (entry["object_id"], entry["frame"])
        if key not in node_lookup:
            node_lookup[key] = entry
    
    for entry in tqdm(json_data, desc="Edge attribute extraction"):
        src_id = entry["object_id"]
        src_frame = entry["frame"]
        src_key = (src_id, src_frame)
        
        if src_key not in node_id_map:
            continue
            
        src_idx = node_id_map[src_key]
        src_bbox = entry["bbox"]
        src_speed = entry["speed"]
        src_direction = entry["direction"]
        src_accel = entry["acceleration"]
        
        # Process neighbors_ids and neighbors_distances together
        if "neighbors_ids" in entry and "neighbors_distances" in entry:
            neighbors_ids = entry["neighbors_ids"]
            neighbors_distances = entry["neighbors_distances"]
            
            # Create valid neighbors as (distance, index, ID) tuples
            valid_neighbors = []
            for i, (nbr_id, distance) in enumerate(zip(neighbors_ids, neighbors_distances)):
                # Skip empty strings or None values
                if not nbr_id or nbr_id == "" or distance is None:
                    continue
                
                try:
                    nbr_id = int(nbr_id)
                    nbr_key = (nbr_id, src_frame)
                    
                    if nbr_key in node_id_map:
                        valid_neighbors.append((distance, i, nbr_id))
                except (ValueError, TypeError):
                    continue
            
            # Sort by distance (ascending order)
            valid_neighbors.sort(key=lambda x: x[0])
            
            # Select only max_neighbors
            for distance, original_idx, nbr_id in valid_neighbors[:max_neighbors]:
                nbr_key = (nbr_id, src_frame)
                dst_idx = node_id_map[nbr_key]
                edge_list.append([src_idx, dst_idx])
                
                if max_edge == 5:
                    # Construct edge attributes (use original_idx to extract information from original data)
                    nbr_speed = entry["neighbors_speeds"][original_idx] if original_idx < len(entry["neighbors_speeds"]) else 0.0
                    rel_speed = [nbr_speed - src_speed]
                    
                    _bbox = [0.0, 0.0, 0.0, 0.0]
                    if nbr_key in node_lookup:
                        nbr_bbox = node_lookup[nbr_key]["bbox"]
                        _bbox = [nbr_bbox[0], nbr_bbox[1], nbr_bbox[2], nbr_bbox[3]]
                    
                    attr = _bbox + rel_speed
                    edge_attrs.append(attr)
                elif max_edge == 6:
                    # Construct edge attributes (use original_idx to extract information from original data)
                    nbr_speed = entry["neighbors_speeds"][original_idx] if original_idx < len(entry["neighbors_speeds"]) else 0.0
                    rel_speed = [nbr_speed - src_speed]
                    
                    nbr_accel = entry["neighbors_accelerations"][original_idx] if original_idx < len(entry["neighbors_accelerations"]) else 0.0
                    rel_accel = [nbr_accel - src_accel]

                    _bbox = [0.0, 0.0, 0.0, 0.0]
                    if nbr_key in node_lookup:
                        nbr_bbox = node_lookup[nbr_key]["bbox"]
                        _bbox = [nbr_bbox[0], nbr_bbox[1], nbr_bbox[2], nbr_bbox[3]]
                    
                    attr = _bbox + rel_speed + rel_accel
                    edge_attrs.append(attr)
                elif max_edge == 7:
                    # Construct edge attributes (use original_idx to extract information from original data)
                    nbr_speed = entry["neighbors_speeds"][original_idx] if original_idx < len(entry["neighbors_speeds"]) else 0.0
                    rel_speed = [nbr_speed - src_speed]
                    
                    nbr_accel = entry["neighbors_accelerations"][original_idx] if original_idx < len(entry["neighbors_accelerations"]) else 0.0
                    rel_accel = [nbr_accel - src_accel]

                    _bbox = [0.0, 0.0, 0.0, 0.0]
                    if nbr_key in node_lookup:
                        nbr_bbox = node_lookup[nbr_key]["bbox"]
                        _bbox = [nbr_bbox[0], nbr_bbox[1], nbr_bbox[2], nbr_bbox[3]]

                    dist = [distance]
                    attr = _bbox + rel_speed + rel_accel +  dist
                    edge_attrs.append(attr)
                elif max_edge == 10:
                    # Construct edge attributes (use original_idx to extract information from original data)
                    nbr_speed = entry["neighbors_speeds"][original_idx] if original_idx < len(entry["neighbors_speeds"]) else 0.0
                    rel_speed = [nbr_speed - src_speed]
                    
                    nbr_accel = entry["neighbors_accelerations"][original_idx] if original_idx < len(entry["neighbors_accelerations"]) else 0.0
                    rel_accel = [nbr_accel - src_accel]

                    _bbox = [0.0, 0.0, 0.0, 0.0]
                    if nbr_key in node_lookup:
                        nbr_bbox = node_lookup[nbr_key]["bbox"]
                        _bbox = [nbr_bbox[0], nbr_bbox[1], nbr_bbox[2], nbr_bbox[3]]

                    nbr_direction = entry["neighbors_directions"][original_idx] if original_idx < len(entry["neighbors_directions"]) else [0.0, 0.0, 0.0]
                    rel_direction = [
                        nbr_direction[0] - src_direction[0],
                        nbr_direction[1] - src_direction[1],
                        nbr_direction[2] - src_direction[2]
                    ]
                    dist = [distance]
                    attr = _bbox + rel_speed + rel_direction + rel_accel + dist
                    edge_attrs.append(attr)
                    
    
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    edge_indices = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Edge attribute extraction complete: {len(edge_attrs)} items (processing time: {elapsed:.2f} seconds)")
    print(f"   Edge attribute dimension: {edge_attr.size(1)}")
    print(f"   Max neighbors: {max_neighbors}")
    
    return edge_attr, edge_indices

def extract_image_data(json_data, node_id_map):
    """
    Extract image paths and bounding box information for each node
    """
    image_paths = []
    bboxes = []
    union_bboxes = []
    
    for entry in json_data:
        # Generate image path for each node (assuming frame_objectid.jpg format)
        frame = entry["frame"]
        obj_id = entry["object_id"]
        image_path = f"frame_{frame}_obj_{obj_id}.jpg"
        
        # Bounding box and union bounding box information
        bbox = entry["bbox"]
        union_bbox = entry.get("union_bbox", bbox)  # Use bbox if union_bbox doesn't exist
        
        image_paths.append(image_path)
        bboxes.append(bbox)
        union_bboxes.append(union_bbox)
    
    return {
        "image_paths": image_paths,
        "bboxes": bboxes,
        "union_bboxes": union_bboxes
    }

def combine_json_data(json_paths):
    """
    Combine data from multiple JSON files into one
    
    Supported JSON structures:
    1. List format: [{"object_id": 1, ...}, {"object_id": 2, ...}]
    2. Dictionary format: {"annotation": [{"object_id": 1, ...}, ...]}
    3. Other structures: Direct data array
    """
    combined_data = []
    
    for json_path in json_paths:
        try:
            with open(json_path, "r", encoding='utf-8') as f:
                print(f"Loading JSON from {json_path}...")
                json_data = json.load(f)
                
                # Extract data according to JSON structure
                if isinstance(json_data, list):
                    # List format: use directly
                    data_to_add = json_data
                    print(f"  Detected list format with {len(json_data)} entries")
                elif isinstance(json_data, dict):
                    # Dictionary format: check for annotation key
                    if 'annotation' in json_data:
                        data_to_add = json_data['annotation']
                        print(f"  Detected dict format with 'annotation' key containing {len(data_to_add)} entries")
                    else:
                        # If annotation key doesn't exist, convert dictionary itself to list
                        data_to_add = [json_data]
                        print(f"  Detected dict format without 'annotation' key, converting to list")
                else:
                    # Other formats: convert single object to list
                    data_to_add = [json_data]
                    print(f"  Detected other format, converting to list")
                
                # Data validation
                if not isinstance(data_to_add, list):
                    print(f"  Warning: Expected list format, got {type(data_to_add)}")
                    data_to_add = [data_to_add] if data_to_add is not None else []
                
                # Check required fields (based on first entry)
                if data_to_add and len(data_to_add) > 0:
                    first_entry = data_to_add[0]
                    required_fields = ['object_id', 'frame', 'bbox', 'category']
                    missing_fields = [field for field in required_fields if field not in first_entry]
                    if missing_fields:
                        print(f"  Warning: Missing required fields in first entry: {missing_fields}")
                
                combined_data.extend(data_to_add)
                print(f"  Added {len(data_to_add)} entries from {json_path}")
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {json_path}: {e}")
        except FileNotFoundError:
            print(f"Error: File not found: {json_path}")
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
    
    print(f"\nCombined data contains {len(combined_data)} total entries")
    
    # Final data validation
    if len(combined_data) == 0:
        print("Warning: No valid data entries found!")
    else:
        print(f"Data structure check:")
        print(f"  First entry keys: {list(combined_data[0].keys())}")
        print(f"  Sample entry: {combined_data[0]}")
    
    return combined_data

def json_to_pyg_data_with_pickle(json_paths, number_neighbor, number_edge, output_dir=None, static_output_file=None):
    """
    Convert multiple JSON data to PyG Data objects
    
    Parameters:
    - json_paths: Single JSON file path or list of JSON file paths
    - cache_path: Cache file path (auto-generated if None)
    - include_images: Whether to include image information
    - save_stats: Whether to save statistics information
    - max_neighbors: Maximum number of neighboring vehicles per vehicle (default: 4)
    """
    # Record initial memory usage
    initial_memory = get_memory_usage()
    print(f"\n🚀 JSON → PyG data conversion starting...")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Step 1: Load and combine JSON data
    start_time = time.time()
    json_data = combine_json_data(json_paths)
    
    elapsed = time.time() - start_time
    memory_usage = get_memory_usage()
    print(f"[1/6] JSON data integration complete: {len(json_data)} entries\n")
    
    # Step 2: Extract node features
    print(f"📊 Node feature extraction starting... (Total {len(json_data)} entries)")

    start_time = time.time()
    # Add tqdm for progress display
    # for i in tqdm(range(0, len(json_data), 1000), desc="Node feature extraction"):
    #     if i + 1000 >= len(json_data):
    #         break
    
    x, node_id_map = extract_node_features(json_data)
    elapsed = time.time() - start_time
    memory_usage = get_memory_usage()
    print(f"✅ Node feature extraction complete: {x.size(0)} nodes, {x.size(1)}-dimensional features (time taken: {elapsed:.2f} seconds)")
    print(f"   Memory usage: {memory_usage:.2f} MB")
    print(f"[2/6] Node feature extraction complete: {x.size(0)} nodes, {x.size(1)} dimensions\n")
    
    # Step 3: Extract and filter labels
    print(f"📋 Label extraction and filtering starting...")
    start_time = time.time()
    # Print categories before filtering
    all_categories = {}
    for entry in json_data:
        cat = entry["category"]
        all_categories[cat] = all_categories.get(cat, 0) + 1
    
    y, filtered_data = extract_labels(json_data)
    elapsed = time.time() - start_time
    print(f"✅ Label extraction complete: {len(y)} items (time taken: {elapsed:.2f} seconds)")
    print(f"[3/6] Label extraction complete: {len(y)} labels\n")
    # Collect results as strings
    stats = []
    for n_neighbor in number_neighbor:
        # Step 4: Extract edge indices
        print(f"🔗 Edge index extraction starting... (neighbor count: {n_neighbor})")
        start_time = time.time()
        edge_index = extract_edge_index(filtered_data, node_id_map, n_neighbor)
        elapsed = time.time() - start_time
        memory_usage = get_memory_usage()
        print(f"✅ Edge index extraction complete: {edge_index.size(1)} edges (time taken: {elapsed:.2f} seconds)")
        print(f"   Memory usage: {memory_usage:.2f} MB")
        print(f"[4/6] Edge index extraction complete: {edge_index.size(1)} edges\n")
        
        for n_edge in number_edge:
            # Step 5: Extract edge attributes
            print(f"🔗 Edge attribute extraction starting...")
            edge_attr, edge_indices = extract_edge_attr(filtered_data, node_id_map, n_neighbor, n_edge)
            elapsed = time.time() - start_time
            memory_usage = get_memory_usage()
            print(f"✅ Edge attribute extraction complete: {len(edge_attr)} edge attributes (time taken: {elapsed:.2f} seconds)")
            print(f"   Memory usage: {memory_usage:.2f} MB")
            print(f"[5/6] Edge attribute extraction complete: {edge_attr.size(0)} edge attributes, {edge_attr.size(1)} dimensions\n")
            # Step 6: Create and save PyG Data objects
            print(f"💾 PyG Data object creation and saving starting...")
            start_time = time.time()
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            elapsed = time.time() - start_time
            final_memory = get_memory_usage()
            print(f"✅ PyG Data object creation and saving complete (time taken: {elapsed:.2f} seconds)")
            print(f"   Final memory usage: {final_memory:.2f} MB (increase: {final_memory - initial_memory:.2f} MB)")
            # 4. Convert selected files to one PKL
            cache_path = os.path.join(output_dir, f"combined_vehicle_data_{n_neighbor}_{n_edge}.pkl")
            print(f"Cache file path: {cache_path}")
            # Save data to cache file using pickle
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[6/6] Data conversion and saving complete: {cache_path}\n")

            # Save additional metadata
            data.num_node_features = x.size(1)
            data.num_edge_features = edge_attr.size(1) if edge_attr.size(0) > 0 else 0
            data.num_classes = len(set([int(label) for label in y]))
            
            # Data summary information
            print(f"📊 Data summary:")
            print(f"   Number of nodes: {data.num_nodes}")
            print(f"   Number of edges: {data.num_edges}")
            print(f"   Node feature dimension: {data.num_node_features}")
            print(f"   Edge feature dimension: {data.num_edge_features}")
            print(f"   Number of categories: {data.num_classes}")
            stats.append("="*50)
            stats.append(f"Number of nodes: {data.num_nodes}")
            stats.append(f"Number of edges: {data.num_edges}")
            stats.append(f"Neighbor count: {n_neighbor}")
            stats.append(f"Node feature dimension: {data.x.size(1)}")
            stats.append(f"Edge feature dimension: {data.edge_attr.size(1) if data.edge_attr.size(0) > 0 else 0}")            
            # Print quantity by category
            if hasattr(data, 'y') and data.y is not None:
                category_counts = {}
                for label in data.y.tolist():
                    category_counts[int(label)] = category_counts.get(int(label), 0) + 1
                print("\n   Distribution by category:")
                stats.append("\n   Distribution by category:")
                total_samples = len(data.y)
                for label, count in sorted(category_counts.items()):
                    percentage = (count / total_samples) * 100
                    # Behavior mapping by label
                    behavior_map = {
                        0: "stop",
                        1: "lane_change", 
                        2: "normal_driving"
                    }
                    behavior = behavior_map.get(label, "unknown")
                    print(f"     Category {behavior}: {count} samples ({percentage:.2f}%)")
                    stats.append(f"     Category {behavior}: {count} samples ({percentage:.2f}%)")
            
            stats.append("="*50)
            # Print to console
            print("\n".join(stats))

    # Save results to file
    with open(static_output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats))
    print(f"\nStatistics saved to: {static_output_file}")

def find_vehicle_json_files(base_dir):
    """
    Find JSON files containing vehicle_data in the given directory and its subdirectories.
    
    Parameters:
    - base_dir: Base directory to start search
    - pattern: Pattern that must be included in file name
    """
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    
    return json_files

# Usage example
if __name__ == "__main__":
    # 1. Single JSON file processing
    # json_path = "/home/lee/research/Research2025/YOLOv11/visualization6/received_file_20240822_101524/json/vehicle_data_updated.json"
    
    # 2. Automatically find JSON files in specific directory
    base_dir = "./2_categorized_vehicle_data"
    json_files = sorted(find_vehicle_json_files(base_dir))
    
    number_neighbor = [4,5,6,7]
    number_edge = [5,6,7,10]
    print(f"Found {len(json_files)} vehicle data JSON files:")
    # for i, path in enumerate(json_files):
    #     print(f"{i+1}. {path}")
    
    # 3. User selects desired files (example using all files here)
    # Check absolute path of currently running file
    current_file_path = os.path.abspath(__file__)
    print(f"Current file path: {current_file_path}")
    print(f"Current directory: {os.path.dirname(current_file_path)}")

    output_folder_name = "4_graph_data"
    # Set path for text file to save results
    output_dir = os.path.join(os.path.dirname(__file__), output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "data_statistics.txt")
    
    json_to_pyg_data_with_pickle(json_files, number_neighbor, number_edge, output_dir=output_dir, static_output_file=output_file)
    