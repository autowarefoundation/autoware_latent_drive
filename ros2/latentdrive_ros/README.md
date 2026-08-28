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

## Live window from the host (X11)

To watch the camera + BEV window on your desktop, forward X11 into the
container (the container's OpenCV is headless, so the window comes from
`rqt_image_view`, which the launch file starts for you).

From the host, in the repo root:

```bash
# 1. allow the container to reach your X server (once per login)
xhost +local:docker

# 2. start the container with X11 + the bag mounted (docker/run.sh does both).
#    BAG is mounted at /bag, the repo at /workspace.
BAG=/path/to/bag_dir ./docker/run.sh
```

Then inside the container shell it drops you into:

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2/install/setup.bash    # colcon build first if never built
ros2 launch latentdrive_ros replay.launch.py \
    bag:=/bag \
    engine:=/workspace/models/engine/e2e-wp10hz-fp16-mixed.engine \
    calibration:=/workspace/sample_data/calibration.json \
    route:=/bag/route.npy
```

An `rqt_image_view` window pops up on your desktop showing the live camera
ribbon + BEV. `display:=false` disables it (headless); `rate:=2.0` plays faster.

Prep the bag once beforehand (see above): `ros2 bag reindex /bag -s sqlite3`
and `export_route.py /bag`.

**If the window doesn't appear:** re-run `xhost +local:docker` on the host and
check `echo $DISPLAY` is non-empty before starting the container. `NO_X11=1`
skips the X11 plumbing entirely.

## Headless check

`scripts/grab_viz.py --count N` saves N frames of `~/viz` as PNGs — useful over
SSH or when no display is available.

## Parameters

`engine_file`, `calibration_file`, `route_file`, `camera_topic`, `odom_topic`,
`accel_topic`, `scene_match` (calibration entry substring, default `prd_jt`),
`subgoal_ahead_m` (default 50), `smooth_alpha` (default 0.35).

`smooth_alpha` is the weight of the fresh plan when smoothing the trajectory
`~/viz` **draws**; `1.0` draws the raw output, lower is smoother and laggier.
The engine is deterministic per frame, but consecutive frames disagree by about
a metre on how far the plan reaches, which makes the BEV tail pulse in and out.
It is display only: `~/plan` always publishes the raw engine output. The filter
carries the previous plan forward by the gap between camera stamps, and resets
itself when that gap goes backwards or exceeds a second, so replaying a looping
bag does not smear a stale plan across the overlay.
