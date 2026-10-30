# Sample 2: rosbag (ROS 2 node, live replay)

Play a vehicle rosbag and let the `autoware_latent_drive` node plan on it as it
plays, the way the vehicle node does: the cameras, odometry, acceleration and
the route points come in as topics, one plan per CAM_FRONT frame goes out,
drawn on the front camera image in a window and, if asked, an mp4.

## Once

```bash
source /opt/ros/<distro>/setup.bash                    # humble or jazzy
scripts/build.sh                                       # or: cd ros2 && colcon build
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
    start_offset:=60 out_video:=/data/bags/1551/replay.mp4
```

`run_replay.sh <bag> <engine> [arg:=value ...]` is
`ros2 launch autoware_latent_drive replay.launch.py` with the paths filled in
(`INSTALL=` points at another colcon install space). It starts the node, waits
8 s for the engine to load and plays the bag with `--clock`. A window shows the
plan on the front camera (Esc or q closes it; Ctrl-C in the terminal stops
everything).

| launch arg | default | |
|---|---|---|
| `camera0`, `camera1`, `camera2` | camera0, camera3, camera4 `/image_raw/compressed` | views 0..2 (CAM_FRONT, FRONT_LEFT, FRONT_RIGHT); a single-camera engine reads camera0 only |
| `missing_view_fill` | `skip` | `zeros`: a side slot with no frame within 50 ms gets the blank frame the camfront3 training used |
| `view_tolerance_s` | `0.05` | max side/front stamp difference per history slot |
| `motion_source` | `pose_difference` | `twist`: velocity/acceleration from the topics instead of differenced poses |
| `max_rate_hz` | `10.0` | plans per second at most, paced on the front stamps |
| `start_offset`, `rate` | `0`, `1.0` | bag position [s] and play rate |
| `show_window` | `true` | `false`: no window (headless; use `out_video`) |
| `out_video` | | also write the camera overlay to this mp4 (at `max_rate_hz` frames per second) |

What a bag needs: the camera topics, `/localization/kinematic_state`,
`/localization/acceleration`, `/tf_static`, the view-0 `camera_info` and
`/tensorrt_e2e/debug/latentdrive/route_points` -- the route points the
vehicle's planner fed its model, which the bags of LatentDrive runs record.
The node takes, per plan, the message of that tick (the vehicle published it
a few tens of ms after the frame), so the model sees exactly the input the
car saw.

## Reading the output

The node logs every 2 s:

```
planning: v=5.5 m/s  |4s|=26.8 m  engine 38.2 ms  (3 views, side views within 14 ms; 227 planned, 0 skipped, 0 superseded, 0 blank slots)
```

- **side views within**: the largest side/front stamp difference used so far
  (14 ms on the prd_jt vehicles: the side cameras fire 14 ms either side of
  the front one).
- **skipped**: plans not run because a side view had no frame within the
  tolerance, or no route points arrived for that tick.
- **superseded**: plans dropped because the next one was ready before the
  engine finished the previous one.

The overlay's ribbon runs green (now) to red (+4 s); the text line gives the
view count, the speed and how far the plan reaches. The raw plan is also
published on `/latentdrive_node/plan` (nav_msgs/Path, base_link).

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
