#!/usr/bin/env python3
"""Export the ego's driven path from a rosbag as route.npy for route_source:=file.

    uv run --with mcap --with mcap-ros2-support --with numpy \\
        python3 export_route.py <bag_dir> [--out route.npy] [--start S] [--end S]

The planner's status needs route points ahead of the ego. The bags of the
LatentDrive runs record the ones the vehicle used
(/tensorrt_e2e/debug/latentdrive/route_points, route_source:=recorded); for
any other bag the ego's own trajectory serves as the route: an open-loop replay
knows the whole run, so "N m of arc length ahead on the driven path" is a
faithful route target wherever the car did not leave its route.

Reads the .mcap files directly (no metadata.yaml or ROS needed).
route.npy: float32 [N, 3] = (map_x, map_y, yaw), one point every 0.5 m.
"""
import argparse
import glob
import os
import sys

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir")
    ap.add_argument("--out", default=None, help="default: <bag_dir>/route.npy")
    ap.add_argument("--topic", default="/localization/kinematic_state")
    ap.add_argument("--start", type=float, default=0.0, help="seconds after the bag start")
    ap.add_argument("--end", type=float, default=None, help="seconds after the bag start")
    ap.add_argument("--spacing", type=float, default=0.5, help="metres between route points")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.bag_dir, "*.mcap")),
                   key=lambda p: int(p[:-5].rsplit("_", 1)[-1]) if p[:-5].rsplit("_", 1)[-1].isdigit() else 0)
    if not files:
        sys.exit(f"no .mcap files in {a.bag_dir}")
    bag_start = None
    for f in files:
        with open(f, "rb") as fh:
            t = make_reader(fh).get_summary().statistics.message_start_time
        bag_start = t if bag_start is None else min(bag_start, t)
    t_from = bag_start + int(a.start * 1e9)
    t_to = None if a.end is None else bag_start + int(a.end * 1e9)

    rows = []
    for f in files:
        with open(f, "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _, _, message, m in reader.iter_decoded_messages(topics=[a.topic]):
                if message.log_time < t_from or (t_to is not None and message.log_time > t_to):
                    continue
                p, q = m.pose.pose.position, m.pose.pose.orientation
                if rows and np.hypot(p.x - rows[-1][0], p.y - rows[-1][1]) < a.spacing:
                    continue
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                rows.append((p.x, p.y, yaw))
    if len(rows) < 2:
        sys.exit(f"{a.topic}: the ego did not move in the selected span")

    route = np.asarray(rows, dtype=np.float32)
    out = a.out or os.path.join(a.bag_dir, "route.npy")
    np.save(out, route)
    length = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum()
    print(f"wrote {out}: {route.shape[0]} points, {length:.1f} m of driven path")


if __name__ == "__main__":
    main()
