#!/usr/bin/env python3
"""Grab frames from the node's /viz image topic and save them as PNGs.

A quick way to inspect the live visualization without a GUI:
    python3 grab_viz.py --topic /latentdrive_node/viz --count 5 --out /workspace/viz
"""
import argparse
import os

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Grabber(Node):
    def __init__(self, topic, count, out):
        super().__init__("grab_viz")
        self.count, self.out, self.n = count, out, 0
        os.makedirs(out, exist_ok=True)
        self.create_subscription(Image, topic, self.cb, qos_profile_sensor_data)

    def cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        rgb = arr[:, :, ::-1]  # bgr8 -> rgb
        path = os.path.join(self.out, f"ros_viz_{self.n:04d}.png")
        PILImage.fromarray(rgb).save(path)
        self.get_logger().info(f"saved {path}")
        self.n += 1
        if self.n >= self.count:
            raise SystemExit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/latentdrive_node/viz")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--out", default="/workspace/viz")
    a = ap.parse_args()
    rclpy.init()
    node = Grabber(a.topic, a.count, a.out)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
