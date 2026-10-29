#!/usr/bin/env python3
"""Cut a sample clip out of a vehicle rosbag (mcap) for run_clip.

    uv run --with mcap --with mcap-ros2-support --with numpy --with opencv-python-headless \\
        python3 export_clip.py <bag_dir> <out_dir> --start 60 --duration 30

Reads the .mcap files of <bag_dir> directly (no metadata.yaml, no ROS
install needed) and writes a self-contained clip:

    clip.json          format, source, time origin, view list with per-frame
                       stamps, the view-0 camera calibration for the overlay
    view0/ view1/ ...  one JPEG per camera frame. By default each frame is
                       already cropped and resized the way the model input is
                       (full width, centred 2:1 band, 512x256), which keeps a
                       30 s three-camera clip near 70 MB; --full-res keeps the
                       recorded JPEG bytes as they are
    odometry.npy       float32 [M, 8]: t, x, y, yaw, vx, vy, ax, ay -- every
                       /localization/kinematic_state message (50 Hz), the
                       acceleration from /localization/acceleration (nearest)
    route.npy          float32 [N, 2]: a route polyline in the map frame. With
                       --lanelet_map, the vehicle's route as its planner built it:
                       the bag's mission-planning route (LaneletRoute) on the
                       lanelet centerlines of scripts/convert_map.py. Without, the
                       ego's driven path from the clip start to --route-ahead
                       seconds past its end, one point every 0.5 m
    route_points.npy   float32 [T, 1 + 2K]: t, x1, y1, ..., xK, yK -- the route
                       points the vehicle's planner fed the model, when the bag
                       recorded /tensorrt_e2e/debug/latentdrive/route_points

Times are seconds since clip.json's t0_ns; map coordinates are relative to
clip.json's origin_xy, so float32 keeps millimetres.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from convert_map import route_polyline  # noqa: E402
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ODOM = "/localization/kinematic_state"
ACCEL = "/localization/acceleration"
ROUTE_POINTS = "/tensorrt_e2e/debug/latentdrive/route_points"
ROUTE = "/planning/mission_planning/route"
TF_STATIC = "/tf_static"
DEFAULT_VIEWS = ["camera0", "camera3", "camera4"]  # CAM_FRONT, FRONT_LEFT, FRONT_RIGHT on prd_jt
VIEW_NAMES = ["front", "front_left", "front_right"]
INPUT_W, INPUT_H = 512, 256


def mcap_files(bag_dir):
    files = glob.glob(os.path.join(bag_dir, "*.mcap"))
    if not files:
        sys.exit(f"no .mcap files in {bag_dir}")

    def index(path):
        stem = os.path.basename(path)[:-5]
        tail = stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0
    return sorted(files, key=index)


def bag_time_range(files):
    start, end = None, None
    for f in files:
        with open(f, "rb") as fh:
            st = make_reader(fh).get_summary().statistics
        start = st.message_start_time if start is None else min(start, st.message_start_time)
        end = st.message_end_time if end is None else max(end, st.message_end_time)
    return start, end


def messages(files, topics, start_ns=None, end_ns=None):
    """Decoded messages of `topics` in log-time order across the split files."""
    for f in files:
        with open(f, "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            st = reader.get_summary().statistics
            if (end_ns is not None and st.message_start_time > end_ns) or \
               (start_ns is not None and st.message_end_time < start_ns):
                continue
            try:  # mcap >= 1.2 renamed the time bounds
                it = reader.iter_decoded_messages(topics=topics, starting_at=start_ns, ending_at=end_ns)
            except TypeError:
                it = reader.iter_decoded_messages(topics=topics, start_time=start_ns, end_time=end_ns)
            for _, channel, message, decoded in it:
                yield channel.topic, message.log_time, decoded


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_in_base_link(transforms, frame):
    """(R, t) of `frame` in base_link by walking the /tf_static tree up."""
    rot, trans = np.eye(3), np.zeros(3)
    seen = set()
    while frame != "base_link":
        if frame not in transforms or frame in seen:
            return None
        seen.add(frame)
        parent, r, t = transforms[frame]
        rot, trans = r @ rot, r @ trans + t
        frame = parent
    return rot, trans


def crop_resize(bgr):
    """The model's preprocessing up to the normalization: full width, centred 2:1 band, 512x256."""
    h, w = bgr.shape[:2]
    band = w // 2
    if h > band:
        top = (h - band) // 2
        bgr = bgr[top:top + band]
    return cv2.resize(bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)


def mission_route(files, bag_start, t0, lanelet_map):
    """The route in effect at the clip start (the last LaneletRoute at or
    before it, else the first after) on the lanelet centerlines, map frame"""
    route = None
    for _, log_time, msg in messages(files, [ROUTE], bag_start):
        if route is not None and log_time > t0:
            break
        route = msg
    if route is None:
        sys.exit(f"--lanelet_map: no {ROUTE} in the bag")
    with open(lanelet_map) as fh:
        centerlines = json.load(fh)["lanelets"]
    ids = [s.preferred_primitive.id for s in route.segments]
    poly, missing = route_polyline(centerlines, ids, (route.goal_pose.position.x, route.goal_pose.position.y))
    if len(poly) < 2:
        sys.exit(f"none of the route's {len(ids)} lanelets is in {lanelet_map}: the route was planned on another map")
    if missing:
        print(f"WARNING: {len(missing)} of the route's {len(ids)} lanelets are not in {lanelet_map} (first: "
              f"{missing[0]}): planned on another map revision; those stretches are bridged with straight lines")
    return poly


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--start", type=float, default=0.0, help="seconds after the bag start")
    ap.add_argument("--duration", type=float, default=30.0, help="clip length [s]")
    ap.add_argument("--views", default=",".join(DEFAULT_VIEWS),
                    help="camera names (or full topics), view 0 first; default camera0,camera3,camera4")
    ap.add_argument("--route-ahead", type=float, default=30.0,
                    help="seconds of driven path past the clip end kept in route.npy")
    ap.add_argument("--lanelet_map", default=None,
                    help="lanelet centerlines (scripts/convert_map.py): route.npy from the bag's "
                         "mission-planning route instead of the driven path")
    ap.add_argument("--full-res", action="store_true", help="keep the recorded JPEGs instead of 512x256 crops")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    a = ap.parse_args()

    files = mcap_files(a.bag_dir)
    bag_start, bag_end = bag_time_range(files)
    t0 = bag_start + int(a.start * 1e9)
    t1 = min(bag_end, t0 + int(a.duration * 1e9))
    # 2 s of history before the clip: the first plan needs 1.5 s of frames and
    # 0.3 s of poses, and the frames are kept so it can run on the clip's first frame
    pre = int(2.0 * 1e9)
    route_end = min(bag_end, t1 + int(a.route_ahead * 1e9))
    print(f"bag {(bag_end - bag_start) / 1e9:.1f} s; clip {a.start:.1f} .. {(t1 - bag_start) / 1e9:.1f} s")

    views = [v if v.startswith("/") else f"/sensing/camera/{v}/image_raw/compressed" for v in a.views.split(",")]
    info_topic = views[0].rsplit("/image_raw", 1)[0] + "/camera_info"
    os.makedirs(a.out_dir, exist_ok=True)
    for i in range(len(views)):
        os.makedirs(os.path.join(a.out_dir, f"view{i}"), exist_ok=True)

    # --- calibration: /tf_static is latched at the bag start, camera_info anywhere
    transforms, info = {}, None
    for topic, _, msg in messages(files, [TF_STATIC, info_topic], bag_start, bag_start + int(5e9)):
        if topic == TF_STATIC:
            for tr in msg.transforms:
                t = tr.transform.translation
                transforms[tr.child_frame_id] = (tr.header.frame_id, quat_to_matrix(tr.transform.rotation),
                                                 np.array([t.x, t.y, t.z]))
        elif info is None:
            info = msg
    camera = None
    if info is not None:
        pose = camera_in_base_link(transforms, info.header.frame_id)
        if pose is not None:
            k = np.array(info.k, dtype=float).reshape(3, 3)
            scale = INPUT_W / info.width
            top = max(0.0, (info.height - info.width / 2) / 2)
            k_crop = k * scale
            k_crop[1, 2] = (k[1, 2] - top) * scale
            k_crop[2] = [0, 0, 1]
            camera = {
                "frame_id": info.header.frame_id,
                "image_size_original": [info.width, info.height],
                "K_crop_resized": k_crop.tolist(),
                "extrinsics_camera_to_ego": {"rotation_matrix": pose[0].tolist(),
                                             "translation": pose[1].tolist()},
            }
    print("view-0 calibration:", "from camera_info + /tf_static" if camera else "not found (no overlay)")

    # --- one pass over the clip window: frames, odometry, acceleration, route points
    frames = [[] for _ in views]
    odom, accel, rpoints = [], [], []
    topics = views + [ODOM, ACCEL, ROUTE_POINTS]
    n = 0
    for topic, _, msg in messages(files, topics, t0 - pre, route_end):
        t = stamp_ns(msg.header)
        if topic == ODOM:
            p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
            yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            odom.append((t, p.x, p.y, yaw, v.x, v.y))
        elif topic == ACCEL:
            accel.append((t, msg.accel.accel.linear.x, msg.accel.accel.linear.y))
        elif topic == ROUTE_POINTS:
            if t0 - pre <= t <= t1:
                rpoints.append((t, [(pt.position.x, pt.position.y) for pt in msg.poses]))
        elif t <= t1:
            v = views.index(topic)
            name = f"{len(frames[v]):06d}.jpg"
            path = os.path.join(a.out_dir, f"view{v}", name)
            if a.full_res:
                with open(path, "wb") as fh:
                    fh.write(bytes(msg.data))
            else:
                bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(path, crop_resize(bgr), [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
            frames[v].append((t, name))
            n += 1
            if n % 300 == 0:
                print(f"  {n} frames", flush=True)
    if not odom:
        sys.exit(f"no {ODOM} in the clip window")
    for v, topic in enumerate(views):
        print(f"view {v} {topic}: {len(frames[v])} frames")

    origin = (odom[0][1], odom[0][2])
    rel = lambda t: (t - t0) / 1e9  # noqa: E731

    # odometry with the nearest acceleration sample
    acc_t = np.array([s[0] for s in accel]) if accel else np.zeros(0)
    rows = []
    for t, x, y, yaw, vx, vy in odom:
        if t > t1:
            break
        ax = ay = 0.0
        if len(acc_t):
            j = int(np.argmin(np.abs(acc_t - t)))
            ax, ay = accel[j][1], accel[j][2]
        rows.append((rel(t), x - origin[0], y - origin[1], yaw, vx, vy, ax, ay))
    np.save(os.path.join(a.out_dir, "odometry.npy"), np.asarray(rows, dtype=np.float32))

    if a.lanelet_map:
        path, route_kind = mission_route(files, bag_start, t0, a.lanelet_map), "mission"
        path = [(x - origin[0], y - origin[1]) for x, y in path]
    else:
        # driven path from the clip start on, one point every 0.5 m
        path, route_kind = [], "driven"
        for t, x, y, *_ in odom:
            if t < t0:
                continue
            p = (x - origin[0], y - origin[1])
            if not path or np.hypot(p[0] - path[-1][0], p[1] - path[-1][1]) >= 0.5:
                path.append(p)
    np.save(os.path.join(a.out_dir, "route.npy"), np.asarray(path, dtype=np.float32))
    length = float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1))) if len(path) > 1 else 0.0
    print(f"route.npy: {route_kind} route, {len(path)} points, {length:.0f} m")

    route_points_file = None
    if rpoints:
        k = len(rpoints[0][1])
        rp = [[rel(t)] + [c for x, y in pts for c in (x - origin[0], y - origin[1])]
              for t, pts in rpoints if len(pts) == k]
        np.save(os.path.join(a.out_dir, "route_points.npy"), np.asarray(rp, dtype=np.float32))
        route_points_file = "route_points.npy"
        print(f"route_points.npy: {len(rp)} recorded ticks x {k} points")
    else:
        print(f"no {ROUTE_POINTS} in the bag: run_clip will build the route points from route.npy")

    clip = {
        "format": "latentdrive-clip-v1",
        "source": os.path.abspath(a.bag_dir),
        "t0_ns": t0,
        "clip_start_in_bag_s": a.start,
        "duration_s": (t1 - t0) / 1e9,
        "origin_xy": list(origin),
        "frames_preprocessed": not a.full_res,
        "views": [{"name": VIEW_NAMES[i] if i < len(VIEW_NAMES) else f"view{i}", "topic": topic,
                   "dir": f"view{i}", "frames": [[rel(t), name] for t, name in frames[i]]}
                  for i, topic in enumerate(views)],
        "camera": camera,
        "odometry": "odometry.npy",
        "route": "route.npy",
        "route_kind": route_kind,
        "route_points": route_points_file,
    }
    with open(os.path.join(a.out_dir, "clip.json"), "w") as fh:
        json.dump(clip, fh, indent=1)
    print(f"wrote {a.out_dir}/clip.json")


if __name__ == "__main__":
    main()
