# Vehicle Behavior Classification Dataset

This repository contains a comprehensive dataset and processing pipeline for analyzing vehicle driving behaviors from video data. The dataset includes lane annotations, vehicle detection results, and driving behavior classifications for research purposes.

## Dataset Overview

The dataset consists of processed video data with vehicle trajectories, lane information, and driving behavior classifications. It is designed for training and evaluating graph neural networks for driving behavior prediction.

## Visual Examples

| Lane Annotations | Vehicle Behavior Visualization |
|:-----------------:|:-------------------------------:|
| <img src="lane_annotations_result.jpg" width="400" height="300"> | <a href="https://www.youtube.com/watch?v=ZOX_vPBZQx4"><img src="https://img.youtube.com/vi/ZOX_vPBZQx4/0.jpg" width="400" height="300"></a> |
|*Interactive lane annotation tool showing road lanes, centerlines, and exclusion zones*|*Annotated video showing vehicle trajectories, behavior classifications, and lane information*|

## Ulsan Gongeoptap Digital Twin Project

<table>
<tr>
  <td colspan="2" align="center">Digital Twin Visualization</td>
</tr>
<tr>
  <td align="center">
    <a href="https://www.youtube.com/watch?v=ybO6N5v70Mc">
      <img src="https://img.youtube.com/vi/ybO6N5v70Mc/0.jpg" width="800" height="600">
    </a>
  </td>
</tr>
<tr>
  <td colspan="2" align="center">Left: Real video clip / Right: Digital Twin</td>
</tr>
</table>

## Processing Pipeline Overview
```
Input Video → Vehicle Detection → Lane Assignment → Behavior Classification → Graph Data
```

## Download Links

### Dataset Files
- [json_data (Google Drive)](https://drive.google.com/drive/folders/1F3xJZBhCIh1NUgzfxI0JdAk0mppb1Twk?usp=sharing) - Vehicle detection data
- [video_data (Google Drive)](https://drive.google.com/drive/folders/1eNYsHGSMzCBMo162OBR4L1EE2EC8bsgE?usp=sharing) - Video data
- [graph_data (Google Drive)](https://drive.google.com/drive/folders/1m7u_9K9q3s5-zPCxdXg6gT4iBLGHYcdi?usp=sharing) - Graph data (.pkl)
- [world_graph_data (Google Drive)](https://drive.google.com/drive/folders/1bz7fisGKwMBUmbzaFgpFabr54Ze6ZGr_?usp=sharing) - World graph data (.pkl)
## Folder Structure

```
tii_graph_data/
├── 1_road_roi.py                           # Lane annotation tool
├── 2_exclude_bbox_first.py                 # Vehicle filtering and lane assignment
├── 3_driving_behavior_classification.py    # Driving behavior classification
├── 4_visualization_filtered.py             # Visualization tool
├── 5_convert_relative_image_graphdata.py   # Graph data conversion (image coordinates)
├── 6_convert_relative_world_graphdata.py   # Graph data conversion (world coordinates)
├── road_input.jpg                          # Input road image
├── lane_annotations.json                   # Lane annotation data
├── video_data/                             # Original video files
│   └── received_file_20240822_101524.avi
├── json_data/                              # Raw vehicle detection data
│   └── received_file_20240822_101524.json
│   └── ...
├── 1_filtered_vehicle_data/                # Filtered vehicle data
│   └── received_file_20240822_101524.json
│   └── ...
├── 2_categorized_vehicle_data/             # Behavior classified data
│   └── received_file_20240822_101524.json
│   └── ...
├── 3_visualization_filtered_result/        # Visualization outputs
│   └── received_file_20240822_101524_visualization.mp4
├── 4_graph_data/                           # Graph neural network data (image coordinates)
│   ├── combined_vehicle_data_*.pkl         # PyTorch Geometric data files
│   └── data_statistics.txt                 # Dataset statistics
└── 5_world_graph_data/                     # Graph neural network data (world coordinates)
    ├── combined_vehicle_data_*.pkl         # PyTorch Geometric data files
    └── data_statistics.txt                 # Dataset statistics
```

## File Descriptions

### 1. Lane Annotation Tool (`1_road_roi.py`)
**Purpose**: Interactive tool for annotating road lanes and exclusion zones
**Input**: Road image (`road_input.jpg`)
**Output**: `lane_annotations.json`
**Usage**:
```bash
uv run python 1_road_roi.py --image ./road_input.jpg --json ./lane_annotations.json
```

**User Modifications**:
- Adjust `--display-width` and `--display-height` for different screen resolutions
- Modify lane colors in `get_color()` function
- Add custom exclusion zones using 't' key

### 2. Vehicle Filtering (`2_exclude_bbox_first.py`)
**Purpose**: Filters vehicles based on exclusion zones and assigns lane IDs
**Input**: Raw vehicle detection JSON, lane annotations
**Output**: Filtered vehicle data in `1_filtered_vehicle_data/`
**Usage**:
```bash
uv run python 2_exclude_bbox_first.py --video ./video_data/video.avi --json ./json_data/data.json
```

### 3. Behavior Classification (`3_driving_behavior_classification.py`)
**Purpose**: Classifies driving behaviors (stop, lane_change, normal_driving)
**Input**: Filtered vehicle data
**Output**: Behavior classified data in `2_categorized_vehicle_data/`
**Usage**:
```bash
uv run python 3_driving_behavior_classification.py --json ./1_filtered_vehicle_data/data.json
```

**User Modifications**:
- Adjust `--min-speed` threshold for stop detection (default: 2.5 km/h)
- Modify `--sequence-length` for lane change detection (default: 7 frames)
- Add new behavior categories in `classify_driving_behavior()`

### 4. Visualization (`4_visualization_filtered.py`)
**Purpose**: Creates annotated video with vehicle trajectories and behaviors
**Input**: Behavior classified data, original video
**Output**: Visualization video in `3_visualization_filtered_result/`
**Usage**:
```bash
uv run python 4_visualization_filtered.py --video ./video_data/video.avi --json ./2_categorized_vehicle_data/data.json
```

### 5. Graph Data Conversion - Image Coordinates (`5_convert_relative_image_graphdata.py`)
**Purpose**: Converts vehicle data to graph format using image coordinates
**Input**: Behavior classified data
**Output**: PyTorch Geometric data files in `4_graph_data/`
**Usage**:
```bash
uv run python 5_convert_relative_image_graphdata.py
```

### 6. Graph Data Conversion - World Coordinates (`5_convert_relative_world_graphdata.py`)
**Purpose**: Converts vehicle data to graph format using world coordinates
**Input**: Behavior classified data with world coordinates
**Output**: PyTorch Geometric data files in `4_world_graph_data/`
**Usage**:
```bash
uv run python 6_convert_relative_world_graphdata.py
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA 11.8 (for GPU acceleration)
- uv package manager

### Environment Setup
```bash
# Install uv if not already installed
pip install uv

# Install dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Dependencies
- **PyTorch**: 2.0.0+cu118 (CUDA 11.8 support)
- **PyTorch Geometric**: 2.6.1
- **OpenCV**: 4.10.0.84
- **NumPy**: 1.23.0
- **Pandas**: 2.0.3
- **Matplotlib**: 3.7.5
- **SciPy**: 1.10.1
- **tqdm**: 4.66.5
- **psutil**: 5.9.1

## Processing Pipeline

### Step 1: Lane Annotation
```bash
uv run python 1_road_roi.py
```
- Load road image and annotate lanes
- Define exclusion zones
- Save annotations to JSON

### Step 2: Vehicle Filtering
```bash
uv run python 2_exclude_bbox_first.py
```
- Filter vehicles outside exclusion zones
- Assign lane IDs to vehicles
- Track vehicle persistence

### Step 3: Behavior Classification
```bash
uv run python 3_driving_behavior_classification.py
```
- Classify driving behaviors
- Calculate speed and acceleration
- Generate behavior statistics

### Step 4: Visualization
```bash
uv run python 4_visualization_filtered.py
```
- Create annotated video
- Display vehicle trajectories
- Show behavior classifications

### Step 5: Graph Data Generation
```bash
# For image coordinates
uv run python 5_convert_relative_image_graphdata.py

# For world coordinates
uv run python 5_convert_relative_world_graphdata.py
```
- Convert to PyTorch Geometric format
- Generate node and edge features
- Create multiple graph configurations

## Data Structure

### Raw Vehicle Detection Data (json_data/)
Raw vehicle detection results from video processing:

```json
{
  "object_id": 11,
  "bbox": [878, 211, 39, 15],
  "position": [75.1442261, 4.8946166, -55.26317],
  "class": 0,
  "speed": 44.71412015639241,
  "direction": [12.395780499999987, 0, 0.7846370000000036],
  "acceleration": 62.102944661656124,
  "frame": 2,
  "category": "",
  "neighbors_ids": [9, 82, 8, 95, 31, 108, 62],
  "neighbors_distances": [9.51797110122735, 17.42144386082408, ...],
  "neighbors_speeds": [31.162106502292207, 11.221498435015738, ...],
  "neighbors_directions": [[8.630099999999956, 0, 0.6709290000000223], ...],
  "neighbors_accelerations": [43.280703475405836, 15.585414493077414, ...],
  "neighbors_world_positions": [[84.10864, 4.8946166, -52.0645676], ...],
  "neighbors_bbox": [[801, 212, 42, 19], ...]
}
```

**Key Fields:**
- `object_id`: Unique vehicle identifier
- `bbox`: Bounding box [center_x, center_y, width, height] in image coordinates
- `position`: 3D world coordinates [x, y, z]
- `speed`: Vehicle speed in km/h
- `direction`: Movement direction vector
- `acceleration`: Vehicle acceleration
- `frame`: Video frame number
- `neighbors_*`: Information about nearby vehicles

### Processed Behavior Data (2_categorized_vehicle_data/)
Enhanced data with behavior classifications and lane assignments:

```json
{
  "object_id": 103,
  "bbox": [588, 10, 17, 19],
  "position": [127.450912, 4.8946166, -95.86969],
  "class": 0,
  "speed": 0,
  "direction": [0, 0, 0],
  "acceleration": 3.659789832320519,
  "frame": 70,
  "category": "stop",
  "neighbors_ids": [234, 35, 70, 77, 234, 21, 35],
  "neighbors_distances": [3.2357413573003653, 9.074350923291927, ...],
  "neighbors_speeds": [0, 0, 5.838702133981688, ...],
  "neighbors_directions": [[0, 0, 0], [0, 0, 0], ...],
  "neighbors_accelerations": [0, 0, 0.13912568283737606, ...],
  "neighbors_world_positions": [[124.4507, 4.8946166, -97.08161], ...],
  "neighbors_bbox": [[610, 9, 17, 16], ...],
  "lane_id": "17-2"
}
```

**Additional Fields:**
- `category`: Behavior classification ("stop", "lane_change", "normal_driving")
- `lane_id`: Assigned lane identifier

## Graph Neural Network Data

| **Feature Type** | **Dimension** | **Attributes** |
|:---:|:---:|:---|
| Node $\(\mathcal{V}\)$ | 9 | cx, cy, w, h, speed, dx, dy, dz, acceleration |
| Edge $\(\mathcal{E}\)$ | 5 | cx, cy, w, h, speed |
| Edge $\(\mathcal{E}\)$ | 6 | cx, cy, w, h, speed, acceleration |
| Edge $\(\mathcal{E}\)$ | 7 | cx, cy, w, h, speed, acceleration, $\mathcal{D}$ |
| Edge $\(\mathcal{E}\)$ | 10 | cx, cy, w, h, speed, dx, dy, dz, acceleration, $\mathcal{D}$ |
| Node $\(\mathcal{V}\)$ | 8 | wx, wy, wz, speed, dx, dy, dz, acceleration |
| Edge $\(\mathcal{E}\)$ | 4 | wx, wy, wz, speed |
| Edge $\(\mathcal{E}\)$ | 5 | wx, wy, wz, speed, acceleration |
| Edge $\(\mathcal{E}\)$ | 6 | wx, wy, wz, speed, acceleration, $\mathcal{D}$ |
| Edge $\(\mathcal{E}\)$ | 9 | wx, wy, wz, speed, dx, dy, dz, acceleration, $\mathcal{D}$ |

#### Image Coordinates Graph Data (4_graph_data/)
PyTorch Geometric graph data using image coordinates:

**File Naming Convention:**
- `combined_vehicle_data_{neighbors}_{edges}.pkl`
- Example: `combined_vehicle_data_6_7.pkl` (6 neighbors, 7 edge features)

**Dataset Statistics:**
- **Total Nodes**: 996,122
- **Total Edges**: 3,978,137
- **Node Features**: 9-dimensional
- **Edge Features**: 5,6,7,10 dimensional
- **Neighbor Configurations**: 4,5,6,7 neighbor vehicles

**Behavior Distribution:**
- Stop: 401,549 samples (40.31%)
- Lane Change: 205,212 samples (20.60%)
- Normal Driving: 389,361 samples (39.09%)


#### World Coordinates Graph Data (4_world_graph_data/)
PyTorch Geometric graph data using world coordinates:

**File Naming Convention:**
- `combined_vehicle_data_{neighbors}_{edges}.pkl`
- Example: `combined_vehicle_data_7_9.pkl` (7 neighbors, 9 edge features)

**Dataset Statistics:**
- **Total Nodes**: 996,122
- **Total Edges**: 3,978,137
- **Node Features**: 9-dimensional
- **Edge Features**: 4,5,6,9 dimensional
- **Neighbor Configurations**: 4,5,6,7 neighbor vehicles

**Behavior Distribution:**
- Stop: 401,549 samples (40.31%)
- Lane Change: 205,212 samples (20.60%)
- Normal Driving: 389,361 samples (39.09%)

**Key Differences:**
- **Image Coordinates**: Smaller dataset, normalized coordinates (0-1), suitable for CNN-GNN hybrid models
- **World Coordinates**: Larger dataset, real-world coordinates, suitable for pure GNN models


## Output Files

### Graph Data Files
- `combined_vehicle_data_{neighbors}_{edges}.pkl`: PyTorch Geometric Data objects
- `data_statistics.txt`: Dataset statistics and class distribution

### Visualization Files
- `*_visualization.mp4`: Annotated videos with vehicle trajectories
- `lane_annotations_result.jpg`: Lane annotation visualization

## Customization Guide

### Adding New Behaviors
1. Modify `classify_driving_behavior()` in `3_driving_behavior_classification.py`
2. Update `CATEGORY_COLORS` in `4_visualization_filtered.py`
3. Adjust label mapping in graph conversion scripts

### Modifying Graph Features
1. Edit `extract_node_features()` for node feature changes
2. Modify `extract_edge_attr()` for edge feature changes
3. Update feature dimensions in graph conversion scripts

### Changing Coordinate Systems
- Use `5_convert_relative_image_graphdata.py` for image coordinates
- Use `5_convert_relative_world_graphdata.py` for world coordinates
- Modify coordinate transformation functions as needed

## Citation

If you use this dataset in your research, please cite:
```
@dataset{comming soon,
  title={comming soon},
  author={comming soon},
  year={comming soon},
  url={comming soon}
}
```

## License

This dataset is provided for research purposes only. Please ensure compliance with local data protection regulations when using this dataset.
