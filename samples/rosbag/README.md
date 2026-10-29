# Sample 2: rosbag (ROS 2 node, live replay)

Play a vehicle rosbag and let the `latentdrive_ros` node plan on it as it
plays, the way the vehicle node does: the cameras, odometry and acceleration
come in as topics, one plan per CAM_FRONT frame goes out, and the route comes
from mission planning like on the car. RViz shows the plan, the route points
and the camera + BEV image next to the plan the vehicle recorded.

## Once

```bash
source /opt/ros/<distro>/setup.bash                    # humble or jazzy
scripts/build.sh                                       # or: cd ros2 && colcon build
```

The package needs `autoware_planning_msgs` (the route message): in the
Autoware images it is under `/opt/autoware`, which `scripts/build.sh`
sources; elsewhere `apt install ros-<distro>-autoware-planning-msgs`.

Per map: the route the model follows is the centerlines of the route's
lanelets, so the replay needs the lanelet2 map the vehicle drove with (the
one its route was planned on, e.g. from Web.Auto). `convert_map.py` turns
its `lanelet2_map.osm` into those centerlines, computed exactly as
Autoware's map loader does (5 m fine centerlines), as one JSON file:

```bash
scripts/convert_map.py /data/maps/1423-20261021061041533656 /data/maps/1423.json
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
samples/rosbag/run_replay.sh /data/bags/1551 models/engine/camfront3-rptw-lane2-e1-mixed.engine \
    lanelet_map:=/data/maps/1423.json start_offset:=60
```

`run_replay.sh <bag> <engine> [arg:=value ...]` is `ros2 launch latentdrive_ros
replay.launch.py` with the paths filled in (`INSTALL=` points at another
colcon install space). It starts the node, waits 8 s for the engine to load,
plays the bag with `--clock` and opens RViz.

| launch arg | default | |
|---|---|---|
| `lanelet_map` | | the map's centerlines (`scripts/convert_map.py`), for `route_source:=mission` |
| `route_source` | `auto` | `mission`: the bag's `/planning/mission_planning/route` on `lanelet_map`, as the vehicle builds it; `recorded`: the route points the vehicle fed its model (`/tensorrt_e2e/debug/latentdrive/route_points`, only in the bags of LatentDrive runs); `file`: built from `route:=route.npy`; `auto`: `mission` when `lanelet_map` is given, else `recorded` |
| `route` | | for `route_source:=file`; make one with `ros2/latentdrive_ros/scripts/export_route.py <bag_dir>` (the driven path) |
| `camera0`, `camera1`, `camera2` | camera0, camera3, camera4 `/image_raw/compressed` | views 0..2 (CAM_FRONT, FRONT_LEFT, FRONT_RIGHT); a single-camera engine reads camera0 only |
| `missing_view_fill` | `skip` | `zeros`: a side slot with no frame within 50 ms gets the blank frame the camfront3 training used |
| `view_tolerance_s` | `0.05` | max side/front stamp difference per history slot |
| `motion_source` | `pose_difference` | `twist`: velocity/acceleration from the topics instead of differenced poses |
| `max_rate_hz` | `10.0` | plans per second at most, paced on the front stamps |
| `start_offset`, `rate` | `0`, `1.0` | bag position [s] and play rate |
| `viewer` | `rviz` | `rqt` (image only), `none` (headless) |
| `show_route_points` | `false` | `true`: draw the route points fed to the model in the BEV panel (the RViz layer is off by default either way) |
| `calibration` | | a calibration.json for the overlay; by default it comes from the bag's camera_info and `/tf_static` |

What a bag needs: the camera topics, `/localization/kinematic_state`,
`/localization/acceleration`, `/tf_static` and the view-0 `camera_info`, and
`/planning/mission_planning/route` for `route_source:=mission` (any Autoware
bag that recorded it) or the vehicle's route points for
`route_source:=recorded` (LatentDrive runs only).

### The route

With `route_source:=mission` the node takes the route the way the vehicle's
planner does: the route's preferred lanelets in order, their centerlines
joined into one polyline and cut at the goal, then the route points 5 .. 50 m
ahead of the ego along it. The route is published once, when it is set, so
the launch latches the one in effect at `start_offset` from the bag
(`publish_latched.py`); a reroute later in the bag is played as recorded.

On a LatentDrive run (2026-10-23, map 1423) the route points this gives are
the ones the vehicle recorded, to the centimetre, on every tick but the few
where the vehicle's odometry sample came 20 ms later (one 1 m vertex). The
node logs the route it built (`route: 70 lanelets -> 584 centerline points`)
and an error when route lanelets are missing from the map: the route was
planned on another map revision, the stretches without lanelets are bridged
with straight lines, and the route points there are unreliable.

## RViz

The view follows the car from behind and above (ThirdPersonFollower on
`base_link`; a top-down view is saved under Views). With `lanelet_map:=` the
map is drawn under the plan by `publish_map.py`: the road surface, lane lines
(dashed ones dashed), road borders, stop lines and crosswalks, the route's
lanes in blue and a box for the car. Only the map within 250 m of the car and
4 m of its height is drawn, redrawn as it moves: under an elevated road the
chase camera would otherwise look at the road above. The map is drawn 0.3 m
below its elevation, where localization puts the car.

In a container RViz renders on the GPU only when the container may use it for
graphics: `docker run ... --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all`.
Without it (`libGL error: failed to load driver`) RViz falls back to software
rendering at a few frames per second.

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

### Lost frames

`replay.launch.py` points Fast DDS at `config/fastdds_replay.xml`, which keeps
the ~1 MB camera frames on shared memory. Without it they go out as
fragmented UDP, the default 208 KB socket buffer overflows and whole seconds
of frames are lost: the log shows `camera stamp jumped ... s` and most plans
are skipped (70 % on an RTX 4090 desktop, with 32 idle cores). A profile set
in `FASTRTPS_DEFAULT_PROFILES_FILE` before the launch is used instead; if
yours does not enlarge the shared-memory segment, expect the same.

With it, the 09-30 bag plays at `rate:=1.0` with 0 skipped on the desktop
(engine 13.8 ms). On a laptop RTX 4060 / i7-13700H the decodes still fell
behind under the GPU's power draw and 15-20 % of the plans were skipped at
`rate:=1.0`; at `rate:=0.5` none were. The node paces on the bag's stamps, so
a slower playback gives the same plans.
