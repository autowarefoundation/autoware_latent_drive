# latentdrive_ros

Minimal ROS 2 node (humble or jazzy) for **open-loop rosbag replay** of the
LatentDrive planner: play a bag, run the engine live on the cameras + ego
state, publish the plan and draw it on the front camera image -- in a window
and/or an mp4. It links the parent repo's library (`Planner`, `ViewWindow`,
`StatusBuilder`, `CameraProjector`); no Autoware packages are needed. The
step-by-step usage is the rosbag sample,
[`samples/rosbag`](../../samples/rosbag).

The engine sets the contract and the node follows it: 1 view (CAM_FRONT) or 3
(CAM_FRONT, FRONT_LEFT, FRONT_RIGHT; 2 for the older wide + telephoto),
status `[1, 6]` (one subgoal) or `[1, 16]` (six route points), 8 or 40
waypoints.

## How it plans

- **Threads.** Each camera has its own executor thread, which decodes and
  preprocesses its frames and pushes them into the window; a planning thread
  runs the engine; a drawing thread draws the overlay and owns the window. A
  camera callback never waits for an inference (the vehicle node's decode
  thread + tick thread split).
- **Tick.** One plan per view-0 frame, paced on the frame stamps to at most
  `max_rate_hz` (the vehicle node's `tick_trigger:=camera`).
- **Side views.** For each history slot, each side view's frame nearest the
  front frame, within `view_tolerance_s`. The newest slot waits for a side
  frame still in flight, up to the tolerance; a slot that stays empty skips
  the plan or, with `missing_view_fill:=zeros`, gets the training's blank
  frame.
- **Status.** The route points are the ones the vehicle's planner fed its
  model, recorded in the bag (`route_points_topic`, default
  `/tensorrt_e2e/debug/latentdrive/route_points`); the plan waits for the
  tick's message. The ego is the odometry stamped nearest the frame, what the
  vehicle's planner saw (a bag plays the frames ~0.1 s late); motion terms
  from differenced poses (`motion_source:=pose_difference`) or the
  twist/acceleration topics.
- **Overlay.** The plan projected on the ground plane of the front image
  (green = now, red = +4 s), with the intrinsics from the view-0
  `camera_info` and the camera pose from `/tf_static`. Esc or q in the window
  ends the node.

## Topics

| direction | topic | type | note |
|---|---|---|---|
| sub | `camera_topics[0..V-1]` (camera0, camera3, camera4 `image_raw/compressed`) | CompressedImage | views, view 0 first |
| sub | `/localization/kinematic_state` | nav_msgs/Odometry | ego pose + velocity |
| sub | `/localization/acceleration` | AccelWithCovarianceStamped | ego acceleration |
| sub | `/tensorrt_e2e/debug/latentdrive/route_points` | geometry_msgs/PoseArray | the route points fed to the model |
| sub | view-0 `camera_info`, `/tf_static` | | camera overlay |
| pub | `~/plan` | nav_msgs/Path | the raw engine output, base_link |

## Parameters

`engine_file`; `camera_topics` (string array); `view_tolerance_s` (0.05),
`history_tolerance_s` (0.05), `missing_view_fill` (skip | zeros),
`max_rate_hz` (10); `route_points_topic`; `motion_source` (pose_difference |
twist); `odom_topic`, `accel_topic`; `smooth_alpha` (0.35, the drawn plan
only); `show_window` (true), `out_video` (empty: no file), `viz_width` (1024:
the drawn image is `viz_width` x `viz_width / 2`).

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

`scripts/publish_tf_static.py <bag_dir>` latches the bag's `/tf_static` for a
replay that starts past it (the launch runs it).

## Window from a container (X11)

`xhost +local:docker` on the host once per login, then start the container
with X11 forwarded (`BAG=/path/to/bag ./docker/run.sh` does both; the bag is
mounted at `/bag`). If no window appears, re-run `xhost +local:docker` and
check that `echo $DISPLAY` is non-empty before starting the container;
`NO_X11=1` skips the X11 plumbing. Without a display, run with
`show_window:=false out_video:=/path/to.mp4` and watch the file.
