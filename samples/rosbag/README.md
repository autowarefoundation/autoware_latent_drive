# Sample 2: rosbag (ROS 2 node, live replay)

Play a vehicle rosbag and let the `latentdrive_ros` node plan on it as it
plays, the way the vehicle node does: the cameras, odometry and acceleration
come in as topics, one plan per CAM_FRONT frame goes out. RViz shows the plan,
the route points and the camera + BEV image next to the plan the vehicle
recorded.

## Once

```bash
source /opt/ros/<distro>/setup.bash                    # humble or jazzy
cd ros2 && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release && cd ..
```

Per bag: the vehicle records a directory of `.mcap` files without
`metadata.yaml`. `prepare_bag.sh` makes it playable without touching it (a
directory of symlinks plus the `metadata.yaml` that `ros2 bag reindex`
writes; on humble the jazzy QoS strings it cannot parse are blanked):

```bash
samples/rosbag/prepare_bag.sh /media/.../0ed0b1e4-..._2026-10-23-15-51-20_p0900 /data/bags/1551
```

## Run

```bash
samples/rosbag/run_replay.sh /data/bags/1551 models/engine/camfront3-rptw-lane2-e1-mixed.engine start_offset:=60
```

`run_replay.sh <bag> <engine> [arg:=value ...]` is `ros2 launch latentdrive_ros
replay.launch.py` with the paths filled in (`INSTALL=` points at another
colcon install space). It starts the node, waits 8 s for the engine to load,
plays the bag with `--clock` and opens RViz.

| launch arg | default | |
|---|---|---|
| `route_source` | `recorded` | `recorded`: the route points the vehicle fed its model (`/tensorrt_e2e/debug/latentdrive/route_points`, in the bags of LatentDrive runs); `file`: build them from `route:=route.npy` |
| `route` | | for `route_source:=file`; make one with `ros2/latentdrive_ros/scripts/export_route.py <bag_dir>` (the driven path) |
| `camera0`, `camera1`, `camera2` | camera0, camera3, camera4 `/image_raw/compressed` | views 0..2 (CAM_FRONT, FRONT_LEFT, FRONT_RIGHT); a single-camera engine reads camera0 only |
| `missing_view_fill` | `skip` | `zeros`: a side slot with no frame within 50 ms gets the blank frame the camfront3 training used |
| `view_tolerance_s` | `0.05` | max side/front stamp difference per history slot |
| `motion_source` | `pose_difference` | `twist`: velocity/acceleration from the topics instead of differenced poses |
| `max_rate_hz` | `10.0` | plans per second at most, paced on the front stamps |
| `start_offset`, `rate` | `0`, `1.0` | bag position [s] and play rate |
| `viewer` | `rviz` | `rqt` (image only), `none` (headless) |
| `calibration` | | a calibration.json for the overlay; by default it comes from the bag's camera_info and `/tf_static` |

What a bag needs: the camera topics, `/localization/kinematic_state`,
`/localization/acceleration`, `/tf_static` and the view-0 `camera_info`, and
for `route_source:=recorded` the vehicle's route points (only LatentDrive runs
have them; for any other bag use `route_source:=file`).

## Reading the output

The node logs every 2 s:

```
planning: v=5.5 m/s  |4s|=26.8 m  engine 38.2 ms  (3 views, side views within 14 ms; 227 planned, 0 skipped, 0 superseded, 0 blank slots)
```

- **side views within**: the largest side/front stamp difference used so far
  (14 ms on the prd_jt vehicles: the side cameras fire 14 ms either side of
  the front one).
- **skipped**: plans not run because a side view had no frame within the
  tolerance (or, in recorded mode, no route points for that tick).
- **superseded**: plans dropped because the next one was ready before the
  engine finished the previous one.

Topics: `/latentdrive_node/plan` (nav_msgs/Path, base_link, the raw engine
output), `/latentdrive_node/plan_map` and `/latentdrive_node/route_points`
(map frame), `/latentdrive_node/viz/compressed` (camera ribbon + BEV).

### On a laptop

Three 2880x1860 JPEG streams at 10 Hz plus the engine are a desktop load.
On a laptop RTX 4060 / i7-13700H the decodes fell behind under the GPU's power
draw and 15-20 % of the plans were skipped at `rate:=1.0`; at `rate:=0.5`
none were. The node paces on the bag's stamps, so a slower playback gives the
same plans. The vehicle's own node (autoware_tensorrt_e2e) holds 10 Hz with
none skipped on the vehicle computer.
