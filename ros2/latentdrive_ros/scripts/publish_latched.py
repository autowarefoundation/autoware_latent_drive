#!/usr/bin/env python3
"""Publish a rosbag's latched topics, latched, for a replay that starts late.

    publish_latched.py <bag_dir> [--start-offset 0] [--route-topic /planning/mission_planning/route]

/tf_static and the mission-planning route are published once, when they
change, so `ros2 bag play --start-offset` skips them, and without them nothing
can place the cameras in base_link or know the route. This publishes, with
transient-local durability:

- /tf_static: the static transforms of the first seconds of the bag (not the
  whole bag: that would decompress every chunk), merged into one message;
- the route (autoware_planning_msgs/LaneletRoute): the last one recorded at or
  before the playback start, or the first one after it. Routes recorded later
  (a reroute) are played by the bag itself.

Then it idles until shut down so late subscribers still get them.
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
from rosidl_runtime_py.utilities import get_message
from tf2_msgs.msg import TFMessage


def open_bag(bag_dir, info, topic):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=info["storage_identifier"]),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    return reader


def read_static(bag_dir, info, seconds):
    reader = open_bag(bag_dir, info, "/tf_static")
    start_ns = info["starting_time"]["nanoseconds_since_epoch"]
    transforms = {}
    while reader.has_next():
        _, data, t = reader.read_next()
        if t > start_ns + seconds * 1e9:
            break
        for tr in deserialize_message(data, TFMessage).transforms:
            transforms[tr.child_frame_id] = tr
    return list(transforms.values())


def read_route(bag_dir, info, topic, start_offset):
    """(type name, serialized route) in effect at the playback start, or None"""
    types = {t["topic_metadata"]["name"]: t["topic_metadata"]["type"] for t in info["topics_with_message_count"]}
    if topic not in types:
        return None
    reader = open_bag(bag_dir, info, topic)
    play_ns = info["starting_time"]["nanoseconds_since_epoch"] + start_offset * 1e9
    route = None
    while reader.has_next():
        _, data, t = reader.read_next()
        if route is not None and t > play_ns:
            break
        route = data
    return None if route is None else (types[topic], route)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir")
    ap.add_argument("--seconds", type=float, default=5.0, help="read /tf_static from this much of the bag start")
    ap.add_argument("--start-offset", type=float, default=0.0, help="where the playback starts in the bag [s]")
    ap.add_argument("--route-topic", default="/planning/mission_planning/route", help="empty: no route")
    a, ros_args = ap.parse_known_args()
    # Own signal handling: launch and the terminal may both deliver a SIGINT,
    # and rclpy's handler turns the second one into a traceback on the way out
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    rclpy.init(args=ros_args, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    node = rclpy.create_node("latched_from_bag")
    with open(os.path.join(a.bag_dir, "metadata.yaml")) as f:
        info = yaml.safe_load(f)["rosbag2_bagfile_information"]
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)

    transforms = read_static(a.bag_dir, info, a.seconds)
    node.create_publisher(TFMessage, "/tf_static", qos).publish(TFMessage(transforms=transforms))
    node.get_logger().info(f"latched {len(transforms)} static transforms from {a.bag_dir}")

    route = read_route(a.bag_dir, info, a.route_topic, a.start_offset) if a.route_topic else None
    if route is not None:
        msg_type = get_message(route[0])
        msg = deserialize_message(route[1], msg_type)
        node.create_publisher(msg_type, a.route_topic, qos).publish(msg)
        node.get_logger().info(f"latched the route ({len(msg.segments)} lanelets) on {a.route_topic}")
    elif a.route_topic:
        node.get_logger().info(f"no {a.route_topic} in the bag")

    # The DDS writers serve late subscribers on their own; nothing to spin
    stop.wait()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
