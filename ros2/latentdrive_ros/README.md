# latentdrive_ros

Minimal ROS 2 node (humble or jazzy) for **open-loop rosbag replay** of the
LatentDrive planner: play a bag, run the engine live on the cameras + ego
state, publish the plan and a camera-overlay + BEV image. It links the parent
repo's library (`Planner`, `ViewWindow`, `StatusBuilder`, the visualizers);
no Autoware packages are needed. The step-by-step usage is the rosbag sample,
[`samples/rosbag`](../../samples/rosbag).

The engine sets the contract and the node follows it: 1 view (CAM_FRONT) or 3
(CAM_FRONT, FRONT_LEFT, FRONT_RIGHT; 2 for the older wide + telephoto),
status `[1, 6]` (one subgoal) or `[1, 16]` (six route points), 8 or 40
waypoints.

## How it plans

- **Threads.** Each camera has its own executor thread, which decodes and
  preprocesses its frames and pushes them into the window; a planning thread
  runs the engine; a drawing thread renders the overlay. A camera callback
  never waits for an inference (the vehicle node's decode thread + tick
  thread split).
- **Tick.** One plan per view-0 frame, paced on the frame stamps to at most
  `max_rate_hz` (the vehicle node's `tick_trigger:=camera`).
- **Side views.** For each history slot, each side view's frame nearest the
  front frame, within `view_tolerance_s`. The newest slot waits for a side
  frame still in flight, up to the tolerance; a slot that stays empty skips
  the plan or, with `missing_view_fill:=zeros`, gets the training's blank
  frame.
- **Status.** Route points from the mission-planning route
  (`route_source:=mission`: the route's lanelet centerlines from
  `lanelet_map`, a JSON of `scripts/convert_map.py`, built like the vehicle
  does), from the vehicle's recording (`route_source:=recorded`, the plan
  waits for the tick's message) or from a map-frame `route.npy`
  (`route_source:=file`). The ego is the odometry stamped nearest the frame,
  what the vehicle's planner saw (a bag plays the frames ~0.1 s late);
  motion terms from differenced poses (`motion_source:=pose_difference`) or
  the twist/acceleration topics.

## Topics

| direction | topic | type | note |
|---|---|---|---|
| sub | `camera_topics[0..V-1]` (camera0, camera3, camera4 `image_raw/compressed`) | CompressedImage | views, view 0 first |
| sub | `/localization/kinematic_state` | nav_msgs/Odometry | ego pose + velocity |
| sub | `/localization/acceleration` | AccelWithCovarianceStamped | ego acceleration |
| sub | `/planning/mission_planning/route` | autoware_planning_msgs/LaneletRoute | `route_source:=mission` only; transient local |
| sub | `/tensorrt_e2e/debug/latentdrive/route_points` | geometry_msgs/PoseArray | `route_source:=recorded` only |
| sub | view-0 `camera_info`, `/tf`, `/tf_static` | | camera overlay, when no `calibration_file` |
| pub | `~/plan` | nav_msgs/Path | the raw engine output, base_link |
| pub | `~/plan_map` | nav_msgs/Path | the same in the map frame (through the ego pose it was planned from) |
| pub | `~/route_points` | geometry_msgs/PoseArray | the route points fed to the model, map frame |
| pub | `~/viz/compressed`, `~/viz` | CompressedImage, Image | camera ribbon + BEV; the raw one only while subscribed |

## Parameters

`engine_file`; `camera_topics` (string array); `view_tolerance_s` (0.05),
`history_tolerance_s` (0.05), `missing_view_fill` (skip | zeros),
`max_rate_hz` (10); `route_source` (mission | recorded | file),
`lanelet_map`, `route_topic`, `route_file`, `recorded_route_points_topic`, `route_points_m` (empty = 5..50 m or 50 m from
the engine), `route_reversal_deg` (120); `motion_source` (pose_difference |
twist); `odom_topic`, `accel_topic`; `calibration_file` + `scene_match`
(overlay from a hand-off calibration.json instead of the bag); `smooth_alpha`
(0.35, the drawn plan only), `viz_jpeg_quality`, `viz_reliable`.

## Build and run

```bash
source /opt/ros/<distro>/setup.bash
cd ros2 && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 launch latentdrive_ros replay.launch.py bag:=/path/to/bag engine:=/path/to.engine
```

`TENSORRT_ROOT` (env or `-DTENSORRT_ROOT`) points at a TensorRT tarball when
it is not installed system-wide. The launch arguments are listed in
[`samples/rosbag/README.md`](../../samples/rosbag/README.md).

Scripts: `export_route.py <bag_dir>` writes the driven path as `route.npy` for
`route_source:=file` (reads the `.mcap` files directly; run it with
`uv run --with mcap --with mcap-ros2-support --with numpy`);
`publish_latched.py <bag_dir> --start-offset S` latches the bag's `/tf_static`
and the route in effect at S for a replay that starts past them (the launch
runs it); `grab_viz.py --count N` saves N
frames of `~/viz` as PNGs (headless check).

## Live window from a container (X11)

`xhost +local:docker` on the host once per login, then start the container
with X11 forwarded (`BAG=/path/to/bag ./docker/run.sh` does both; the bag is
mounted at `/bag`). If no window appears, re-run `xhost +local:docker` and
check that `echo $DISPLAY` is non-empty before starting the container;
`NO_X11=1` skips the X11 plumbing.
