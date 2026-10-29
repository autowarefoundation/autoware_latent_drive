#!/usr/bin/env python3
"""Draw the lanelet map, the route and the ego car for RViz.

    publish_map.py <map.json> [--route-topic /planning/mission_planning/route]
                              [--odom-topic /localization/kinematic_state]

<map.json> is what scripts/convert_map.py writes. Published (transient
local), in the map frame:

- ~/vector_map (MarkerArray): the road surface, the lane lines (dashed ones
  dashed), road borders, stop lines and crosswalks;
- ~/route (MarkerArray): the lanes of the mission-planning route, highlighted;
- ~/ego (Marker): a box the size of the car on base_link.

Only the map around the car is drawn: within --radius of it and within
--height-band of its elevation. A chase camera sits ~13 m above the car, and
under an elevated road the full map puts that road between the camera and the
car. The part drawn follows the car (redrawn every --redraw-m metres).

Under a bag's --clock (use_sim_time:=true) RViz resets its displays whenever
the time jumps -- when the bag starts, at every loop -- and drops what it had
received; latched messages are not delivered again, so everything is
republished after each jump.
"""
import argparse
import json
import math
import signal
import threading

import numpy as np
import rclpy
import rclpy.signals
from autoware_planning_msgs.msg import LaneletRoute
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType, JumpThreshold
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

ROAD = (0.22, 0.24, 0.27, 1.0)
SIDEWALK = (0.32, 0.33, 0.35, 1.0)
CROSSWALK = (0.85, 0.85, 0.85, 0.45)
ROUTE = (0.10, 0.55, 0.95, 0.40)
LINES = {  # kind: (width [m], colour)
    "lane": (0.15, (0.92, 0.92, 0.92, 1.0)),
    "lane_dashed": (0.15, (0.92, 0.92, 0.92, 1.0)),
    "road_border": (0.25, (0.62, 0.64, 0.68, 1.0)),
    "stop_line": (0.45, (0.95, 0.95, 0.95, 1.0)),
}
DASH_M, GAP_M = 3.0, 3.0
# The map is drawn this far below its own elevation: localization puts
# base_link (on the ground) 0.1-0.2 m under the map's `ele`, which would bury
# the plan and the car in the road surface. Layers 5 cm apart, no z-fighting
SINK_M = 0.3
ROAD_DZ, ROUTE_DZ, CROSSWALK_DZ, LINE_DZ = -SINK_M, -SINK_M + 0.05, -SINK_M + 0.10, -SINK_M + 0.15
STRIP_STEP = 2  # every 2nd pair of the 2 m bounds: 4 m triangles
WALK = ("pedestrian_lane", "walkway", "road_shoulder")


class Layer:
    """One marker's primitives: [P, 3, 3] triangles or [P, 2, 3] segments"""

    def __init__(self, ns, kind, rgba, prims, width=1.0):
        self.ns, self.kind, self.rgba, self.width = ns, kind, rgba, width
        self.prims = np.asarray(prims, dtype=np.float64).reshape(-1, 3 if kind == Marker.TRIANGLE_LIST else 2, 3)
        self.center = self.prims.mean(axis=1)

    def marker(self, ego, radius, band):
        m = Marker()
        m.header.frame_id = "map"
        m.ns, m.id, m.type, m.action = self.ns, 0, self.kind, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = self.width
        m.scale.y = m.scale.z = 1.0
        m.color = ColorRGBA(r=self.rgba[0], g=self.rgba[1], b=self.rgba[2], a=self.rgba[3])
        c = self.center
        near = (np.hypot(c[:, 0] - ego[0], c[:, 1] - ego[1]) < radius) & (np.abs(c[:, 2] - ego[2]) < band)
        # One assignment: every assignment to a message's sequence field
        # re-checks the whole list, so growing it piecewise is quadratic
        m.points = [Point(x=x, y=y, z=z) for x, y, z in self.prims[near].reshape(-1, 3).tolist()]
        if not m.points:
            m.action = Marker.DELETE  # RViz rejects an empty list; this clears what was drawn
        return m


def strip(lane, dz):
    """A lane's area as triangles between its paired bounds"""
    idx = list(range(0, len(lane["left"]), STRIP_STEP))
    if idx[-1] != len(lane["left"]) - 1:
        idx.append(len(lane["left"]) - 1)
    left = np.asarray([lane["left"][i] for i in idx]) + [0, 0, dz]
    right = np.asarray([lane["right"][i] for i in idx]) + [0, 0, dz]
    if len(left) < 2:
        return np.zeros((0, 3, 3))
    first = np.stack([left[:-1], right[:-1], left[1:]], axis=1)
    second = np.stack([left[1:], right[:-1], right[1:]], axis=1)
    return np.concatenate([first, second])


def segments(points, dz):
    p = np.asarray(points) + [0, 0, dz]
    return np.stack([p[:-1], p[1:]], axis=1)


def dashes(points, dz):
    """The dash segments along a polyline"""
    out, s, on = [], 0.0, True
    for a, b in zip(points, points[1:]):
        seg = math.dist(a[:2], b[:2])
        t = 0.0
        while seg > 0 and t < seg:
            step = min((DASH_M if on else GAP_M) - s, seg - t)
            if on:
                p = [a[k] + (b[k] - a[k]) * t / seg for k in range(3)]
                q = [a[k] + (b[k] - a[k]) * (t + step) / seg for k in range(3)]
                out.append([[p[0], p[1], p[2] + dz], [q[0], q[1], q[2] + dz]])
            t += step
            s += step
            if s >= (DASH_M if on else GAP_M) - 1e-9:
                s, on = 0.0, not on
    return out


def map_layers(viz):
    areas = {"road": [], "sidewalk": [], "crosswalk": []}
    for lane in viz["lanes"].values():
        if lane["subtype"] == "crosswalk":
            areas["crosswalk"].append(strip(lane, CROSSWALK_DZ))
        elif lane["subtype"] in WALK:
            areas["sidewalk"].append(strip(lane, ROAD_DZ - 0.02))
        else:
            areas["road"].append(strip(lane, ROAD_DZ))
    lines = {kind: [] for kind in LINES}
    for line in viz["lines"]:
        if len(line["points"]) < 2:
            continue
        if line["kind"] == "lane_dashed":
            lines["lane_dashed"] += dashes(line["points"], LINE_DZ)
        else:
            lines[line["kind"]] += segments(line["points"], LINE_DZ).tolist()
    layers = [Layer(ns, Marker.TRIANGLE_LIST, rgba, np.concatenate(areas[ns]) if areas[ns] else [])
              for ns, rgba in (("road", ROAD), ("sidewalk", SIDEWALK), ("crosswalk", CROSSWALK))]
    layers += [Layer(kind, Marker.LINE_LIST, rgba, lines[kind], width) for kind, (width, rgba) in LINES.items()]
    return [layer for layer in layers if len(layer.prims)]


def ego_box(length, width, height, rear_overhang):
    m = Marker()
    m.header.frame_id = "base_link"
    m.ns, m.id, m.type, m.action = "ego", 0, Marker.CUBE, Marker.ADD
    m.frame_locked = True
    m.scale.x, m.scale.y, m.scale.z = length, width, height
    m.pose.position.x = length / 2 - rear_overhang
    m.pose.position.z = height / 2
    m.pose.orientation.w = 1.0
    m.color = ColorRGBA(r=0.85, g=0.87, b=0.92, a=0.9)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_json")
    ap.add_argument("--route-topic", default="/planning/mission_planning/route")
    ap.add_argument("--odom-topic", default="/localization/kinematic_state")
    ap.add_argument("--radius", type=float, default=250.0, help="draw the map this far around the car [m]")
    ap.add_argument("--height-band", type=float, default=4.0,
                    help="draw only what lies within this height of the car [m] (not the road above it)")
    ap.add_argument("--redraw-m", type=float, default=25.0, help="redraw after the car moved this far [m]")
    ap.add_argument("--ego-size", type=float, nargs=4, default=[4.6, 1.8, 1.6, 1.0],
                    metavar=("LENGTH", "WIDTH", "HEIGHT", "REAR_OVERHANG"),
                    help="the ego box [m]; base_link is the rear axle")
    a, ros_args = ap.parse_known_args()
    # Own signal handling, like publish_latched.py: a second SIGINT from the
    # terminal would otherwise end in a traceback
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    rclpy.init(args=ros_args, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    node = rclpy.create_node("map_viz")
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)

    with open(a.map_json) as f:
        viz = json.load(f).get("viz")
    if viz is None:
        node.get_logger().error(f"{a.map_json} has no viz section: convert the map again with convert_map.py")
        return
    layers = map_layers(viz)
    node.get_logger().info(f"vector map: {sum(len(layer.prims) for layer in layers)} primitives from {a.map_json}")
    pub_map = node.create_publisher(MarkerArray, "~/vector_map", latched)
    pub_route = node.create_publisher(MarkerArray, "~/route", latched)
    pub_ego = node.create_publisher(Marker, "~/ego", latched)
    ego_marker = ego_box(*a.ego_size)
    pub_ego.publish(ego_marker)

    state = {"ego": None, "drawn_at": None, "route": None, "map_msg": None, "route_msg": None}

    def on_odom(msg):
        p = msg.pose.pose.position
        state["ego"] = (p.x, p.y, p.z)

    def on_route(msg):
        prims, missing = [], 0
        for segment in msg.segments:
            lane = viz["lanes"].get(str(segment.preferred_primitive.id))
            if lane is None:
                missing += 1
            else:
                prims.append(strip(lane, ROUTE_DZ))
        state["route"] = Layer("route", Marker.TRIANGLE_LIST, ROUTE, np.concatenate(prims) if prims else [])
        state["drawn_at"] = None  # redraw with it
        node.get_logger().info(f"route: {len(msg.segments) - missing} of {len(msg.segments)} lanes drawn")

    def redraw():
        ego, at = state["ego"], state["drawn_at"]
        if ego is None:
            return
        if at is not None and math.hypot(ego[0] - at[0], ego[1] - at[1]) < a.redraw_m and abs(ego[2] - at[2]) < 1.5:
            return
        state["drawn_at"] = ego
        state["map_msg"] = MarkerArray(markers=[layer.marker(ego, a.radius, a.height_band) for layer in layers])
        pub_map.publish(state["map_msg"])
        if state["route"] is not None and len(state["route"].prims):
            state["route_msg"] = MarkerArray(markers=[state["route"].marker(ego, a.radius, a.height_band)])
            pub_route.publish(state["route_msg"])

    def republish():
        pub_ego.publish(ego_marker)
        for pub, key in ((pub_map, "map_msg"), (pub_route, "route_msg")):
            if state[key] is not None:
                pub.publish(state[key])

    node.create_subscription(Odometry, a.odom_topic, on_odom, qos_profile_sensor_data)
    node.create_subscription(LaneletRoute, a.route_topic, on_route, latched)
    # Wall-clock timers: they keep running whatever the bag's clock does
    wall = Clock(clock_type=ClockType.STEADY_TIME)
    node.create_timer(0.5, redraw, clock=wall)
    pending = []

    def on_jump(_):
        # RViz needs a moment after the jump before it listens again
        if not pending:
            pending.append(node.create_timer(1.0, lambda: (republish(), node.destroy_timer(pending.pop())),
                                             clock=wall))

    jump_handle = node.get_clock().create_jump_callback(  # noqa: F841  (kept alive)
        JumpThreshold(min_forward=Duration(seconds=1.0), min_backward=Duration(seconds=-0.1), on_clock_change=True),
        post_callback=on_jump)

    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    stop.wait()
    rclpy.try_shutdown()
    spin.join(timeout=2.0)


if __name__ == "__main__":
    main()
