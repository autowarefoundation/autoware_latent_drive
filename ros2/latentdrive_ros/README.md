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
| pub | `~/viz/compressed` | sensor_msgs/CompressedImage | camera ribbon + BEV panel, JPEG — **the one to view** |
| pub | `~/viz` | sensor_msgs/Image | the same panel raw; only published while something subscribes |
| pub | `~/plan` | nav_msgs/Path | predicted trajectory (base_link) |

### Viewer QoS — read this before wondering why the window is blank

Both `~/viz` topics are **best-effort, keep-last** by default (`SensorDataQoS`),
like the camera topics they mirror: a viewer must never be able to slow the
planner down. A **reliable subscriber will not match them and receives
nothing** — the RMW says so on its own stderr:

```
New publisher discovered on topic '/latentdrive_node/viz', offering
incompatible QoS. No messages will be received from it.
Last incompatible policy: RELIABILITY
```

`rqt_image_view` in Jazzy has no QoS selector and image_transport defaults to
reliable, so **`rqt_image_view` needs `-p viz_reliable:=true`** on the node.
Measured on the 60 s bag, one viewer attached:

| node `viz_reliable` | subscriber | viz rate | dropped-frame resets |
|---|---|---|---|
| `false` (default) | best-effort | 6.6-8.8 Hz | 0 |
| `true` | best-effort | 8.1 Hz | 0 |
| `true` | reliable (rqt_image_view) | 5.7 Hz | 3 |
| `false` | reliable | **nothing** | — |

So: anything that can subscribe best-effort (rclpy `qos_profile_sensor_data`,
rviz2's Image display with Reliability set to Best Effort, `grab_viz.py`) gets
the full camera rate. `rqt_image_view` costs a flag and some frames.

Drawing, JPEG encoding and publishing run on their own thread, fed by a
one-slot mailbox: if a viewer or the encoder falls behind, the pending frame is
replaced rather than queued, and the image callback never waits. That matters
for more than smoothness — a blocked callback drops camera frames, and dropped
frames put the wrong spacing into the 4-frame history the model reads.

Measured on a 60 s bag (10 Hz camera, RTX-class GPU, viewer attached):

| | before | after |
|---|---|---|
| viz message size | 2.41 MB | 122 kB |
| `~/viz` rate | 4.1 Hz, gaps to 1.3 s | 8.8 Hz, no gaps |
| `~/plan` rate | 5.7 Hz, gaps to 1.7 s | 8.1 Hz, gaps to 0.7 s |
| dropped-frame resets | 1-5 per run | 0 |

Image callback budget: 39 ms median of the 100 ms period (JPEG decode of the
2880x1860 frame 33, preprocess 1, inference 5). The viz thread takes 6 ms.

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

# view it -- rqt_image_view subscribes reliably, so start the node with
# -p viz_reliable:=true (see "Viewer QoS" above)
ros2 run rqt_image_view rqt_image_view /latentdrive_node/viz/compressed
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
`subgoal_ahead_m` (default 50), `smooth_alpha` (default 0.35),
`viz_jpeg_quality` (default 80), `viz_reliable` (default false — see
**Viewer QoS** above).

`smooth_alpha` is the weight of the fresh plan when smoothing the trajectory
`~/viz` **draws**; `1.0` draws the raw output, lower is smoother and laggier.
The engine is deterministic per frame, but consecutive frames disagree by about
a metre on how far the plan reaches, which makes the BEV tail pulse in and out.
It is display only: `~/plan` always publishes the raw engine output. The filter
carries the previous plan forward by the gap between camera stamps, and resets
itself when that gap goes backwards or exceeds a second, so replaying a looping
bag does not smear a stale plan across the overlay.
