#!/usr/bin/env python3
"""Publish a rosbag's /tf_static, latched, for a replay that starts late.

    publish_tf_static.py <bag_dir> [--seconds 5]

/tf_static is recorded once, when the bag starts, so `ros2 bag play
--start-offset` never plays it, and without it nothing can place the cameras
in base_link. This reads the static transforms from the first seconds of the
bag (not the whole bag: that would decompress every chunk), merges them into
one message and publishes it with transient-local durability, then idles until
shut down so late subscribers still get it.
"""
import argparse
import os
import signal
import threading

import rclpy
import rclpy.signals
import rosbag2_py
import yaml
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage


def read_static(bag_dir, seconds):
    with open(os.path.join(bag_dir, "metadata.yaml")) as f:
        info = yaml.safe_load(f)["rosbag2_bagfile_information"]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=info["storage_identifier"]),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/tf_static"]))
    start_ns = info["starting_time"]["nanoseconds_since_epoch"]
    transforms = {}
    while reader.has_next():
        _, data, t = reader.read_next()
        if t > start_ns + seconds * 1e9:
            break
        for tr in deserialize_message(data, TFMessage).transforms:
            transforms[tr.child_frame_id] = tr
    return list(transforms.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir")
    ap.add_argument("--seconds", type=float, default=5.0, help="read /tf_static from this much of the bag start")
    a, ros_args = ap.parse_known_args()
    # Own signal handling: launch and the terminal may both deliver a SIGINT,
    # and rclpy's handler turns the second one into a traceback on the way out
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    rclpy.init(args=ros_args, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    node = rclpy.create_node("tf_static_from_bag")
    transforms = read_static(a.bag_dir, a.seconds)
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    node.create_publisher(TFMessage, "/tf_static", qos).publish(TFMessage(transforms=transforms))
    node.get_logger().info(f"latched {len(transforms)} static transforms from {a.bag_dir}")
    # The DDS writer serves late subscribers on its own; nothing to spin
    stop.wait()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
