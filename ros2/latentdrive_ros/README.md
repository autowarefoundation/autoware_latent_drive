# latentdrive_ros

Minimal ROS 2 (Jazzy) node for **open-loop rosbag replay** of the LatentDrive
planner: replay a bag, run the engine live on the front camera + ego state, and
publish the plan plus a camera-overlay + BEV visualization.

It links the parent repo's inference/visualizer core (`Planner`, `FrameWindow`,
`BevVisualizer`, `CameraProjector`) — no autoware packages required.

## Topics

| direction | topic | type | note |
|---|---|---|---|
| sub | `/sensing/camera/camera1/image_raw/compressed` | CompressedImage | front camera (CAM_FRONT_WIDE), 10 Hz |
| sub | `/localization/kinematic_state` | nav_msgs/Odometry | ego pose + velocity |
| sub | `/localization/acceleration` | geometry_msgs/AccelWithCovarianceStamped | ego acceleration |
| pub | `~/viz` | sensor_msgs/Image | camera ribbon + BEV panel |
| pub | `~/plan` | nav_msgs/Path | predicted trajectory (base_link) |

The **subgoal** (route point 50 m ahead) is read from `route.npy` — the ego's
own driven path exported from the bag — because the container has no
`autoware_planning_msgs` to parse `/planning/trajectory`. For open-loop replay
the whole run is known, so this is a faithful route target.

## One-time prep (inside the deployment container)

```bash
# 1. the raw .db3 has no metadata.yaml; let ros2 build it
ros2 bag reindex /path/to/bag_dir -s sqlite3

# 2. export the ego route (needs the rosbags python lib; the host venv has it)
python3 export_route.py /path/to/bag_dir          # -> bag_dir/route.npy
```

## Build

Inside the container (`deployment/` image of the parent repo), with the repo
mounted at `/workspace`:

```bash
source /opt/ros/jazzy/setup.bash
cd /workspace/ros2 && colcon build --paths latentdrive_ros
source install/setup.bash
```

## Run

```bash
# terminal A: the node
ros2 run latentdrive_ros latentdrive_node --ros-args \
    -p engine_file:=/workspace/models/engine/e2e-wp10hz-fp16-mixed.engine \
    -p calibration_file:=/workspace/sample_data/calibration.json \
    -p route_file:=/path/to/bag_dir/route.npy

# terminal B: play the bag
ros2 bag play /path/to/bag_dir --topics \
    /sensing/camera/camera1/image_raw/compressed \
    /localization/kinematic_state /localization/acceleration

# view it
ros2 run rqt_image_view rqt_image_view /latentdrive_node/viz
```

Or both together via the launch file:

```bash
ros2 launch latentdrive_ros replay.launch.py \
    bag:=/path/to/bag_dir \
    engine:=/workspace/models/engine/e2e-wp10hz-fp16-mixed.engine \
    calibration:=/workspace/sample_data/calibration.json \
    route:=/path/to/bag_dir/route.npy
```

`scripts/grab_viz.py --count N` saves N frames of `~/viz` as PNGs for a quick
headless check.

## Parameters

`engine_file`, `calibration_file`, `route_file`, `camera_topic`, `odom_topic`,
`accel_topic`, `scene_match` (calibration entry substring, default `prd_jt`),
`subgoal_ahead_m` (default 50).
