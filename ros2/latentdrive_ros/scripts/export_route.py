#!/usr/bin/env python3
"""Export the ego's driven path from a rosbag as route.npy for the replay node.

The container has no autoware_planning_msgs, so /planning/trajectory can't be
parsed. For an open-loop replay the whole run is known anyway, so we use the
ego's own trajectory (from /localization/kinematic_state) as the route: the
node takes the point 50 m of arc length ahead of the current pose as the
subgoal. The bag needs a metadata.yaml first (`ros2 bag reindex <dir> -s sqlite3`).

    python3 export_route.py <bag_dir> [--out route.npy]

route.npy: float32 [N, 3] = (map_x, map_y, yaw), ordered by time.
"""
import argparse
import os
import sys

import numpy as np


def require_metadata(bag_dir):
    # A raw .db3 has no metadata.yaml. Don't hand-write one -- the ROS 2 Jazzy
    # format (version 9, per-topic QoS, a files: block) is easy to get subtly
    # wrong and `ros2 bag play` then rejects it. Let ros2 build it:
    if not os.path.exists(os.path.join(bag_dir, "metadata.yaml")):
        sys.exit(f"no metadata.yaml in {bag_dir}\n"
                 f"generate it first (inside the container):\n"
                 f"  ros2 bag reindex {bag_dir} -s sqlite3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir")
    ap.add_argument("--out", default=None, help="default: <bag_dir>/route.npy")
    ap.add_argument("--topic", default="/localization/kinematic_state")
    a = ap.parse_args()

    require_metadata(a.bag_dir)
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    with Reader(a.bag_dir) as r:
        conns = [c for c in r.connections if c.topic == a.topic]
        if not conns:
            sys.exit(f"topic {a.topic} not in bag")
        for conn, _, raw in r.messages(connections=conns):
            m = ts.deserialize_cdr(raw, conn.msgtype)
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            rows.append((p.x, p.y, yaw))

    route = np.asarray(rows, dtype=np.float32)
    out = a.out or os.path.join(a.bag_dir, "route.npy")
    np.save(out, route)
    seg = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum()
    print(f"wrote {out}: {route.shape}, total path length {seg:.1f} m")


if __name__ == "__main__":
    main()
