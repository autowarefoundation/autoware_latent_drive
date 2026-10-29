#!/usr/bin/env python3
"""Convert a lanelet2 map into the lanelet centerlines LatentDrive builds its route from.

    convert_map.py <lanelet2_map.osm | map_dir> <out.json> [--resolution 5.0]

On the vehicle the route the model follows is the centerlines of the route's
preferred lanelets (autoware_planning_msgs/LaneletRoute, from mission
planning), as Autoware's map loader leaves them: each lanelet's centerline
regenerated every `center_line_resolution` (5 m) by resampling both bounds to
the same number of points and taking the midpoints
(lanelet::utils::generateFineCenterline), or the lanelet's `waypoints` line
where the map has one (use_waypoints). This writes exactly those centerlines,
so a replay needs only this file and the route in the bag -- no lanelet2,
no map loader:

    {"source": ..., "resolution_m": 5.0, "lanelets": {"<id>": [[x, y], ...], ...}}

Coordinates are the map frame: the nodes' local_x / local_y tags, which
Autoware's MGRS maps carry. Pure Python, no dependencies.
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET


def load_osm(path):
    """Nodes (id -> (x, y, z)), ways (id -> node ids) and lanelet relations
    (id -> (left way, right way, waypoints way or None))"""
    nodes, ways, lanelets = {}, {}, {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "node":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if "local_x" not in tags or "local_y" not in tags:
                sys.exit(f"node {el.get('id')} has no local_x/local_y tags: only maps with local "
                         "coordinates (Autoware's MGRS maps) are supported")
            nodes[int(el.get("id"))] = (float(tags["local_x"]), float(tags["local_y"]), float(tags.get("ele", 0.0)))
            el.clear()
        elif el.tag == "way":
            ways[int(el.get("id"))] = [int(n.get("ref")) for n in el.findall("nd")]
            el.clear()
        elif el.tag == "relation":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if tags.get("type") == "lanelet":
                roles = {m.get("role"): int(m.get("ref")) for m in el.findall("member") if m.get("type") == "way"}
                if "left" in roles and "right" in roles:
                    waypoints = int(tags["waypoints"]) if tags.get("waypoints", "").isdigit() else None
                    lanelets[int(el.get("id"))] = (roles["left"], roles["right"], waypoints)
            el.clear()
    return nodes, ways, lanelets


def resample(points, num_segments):
    """lanelet::utils::resamplePoints: num_segments + 1 points at equal
    fractions of the 3D length, interpolated on the segment that holds them"""
    acc = [0.0]
    for a, b in zip(points, points[1:]):
        acc.append(acc[-1] + math.dist(a, b))
    total, n = acc[-1], len(points)
    out = []
    for i in range(num_segments + 1):
        target = i / num_segments * total
        if target < acc[1]:
            j = 0
        elif target > acc[n - 2]:
            j = n - 2
        else:
            j = next(k - 1 for k in range(1, n) if acc[k - 1] <= target <= acc[k])
        seg = acc[j + 1] - acc[j]
        f = (target - acc[j]) / seg if seg > 0 else 0.0
        out.append(tuple(p + (q - p) * f for p, q in zip(points[j], points[j + 1])))
    return out


def fine_centerline(left, right, resolution):
    """lanelet::utils::generateFineCenterline"""
    length = lambda pts: sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))  # noqa: E731
    num_segments = max(math.ceil(max(length(left), length(right)) / resolution), 1)
    return [((l[0] + r[0]) / 2, (l[1] + r[1]) / 2) for l, r in
            zip(resample(left, num_segments), resample(right, num_segments))]


def convert(osm_path, resolution):
    nodes, ways, lanelets = load_osm(osm_path)
    out, from_waypoints, inverted = {}, 0, 0
    for lid, (left_id, right_id, waypoints_id) in lanelets.items():
        if waypoints_id is not None and waypoints_id in ways:
            line = [nodes[n][:2] for n in ways[waypoints_id]]
            from_waypoints += 1
        else:
            left = [nodes[n] for n in ways[left_id]]
            right = [nodes[n] for n in ways[right_id]]
            if len(left) < 2 or len(right) < 2:
                continue
            # A bound shared with the neighbouring lane may be stored against
            # the driving direction (~10 % of the lanelets of a vehicle map);
            # lanelet2 inverts it on load, the left bound keeps its order. Paired
            # as stored, the midpoints would fold onto the middle of the lanelet
            if math.dist(left[0], right[0]) + math.dist(left[-1], right[-1]) > \
               math.dist(left[0], right[-1]) + math.dist(left[-1], right[0]):
                right = right[::-1]
                inverted += 1
            line = fine_centerline(left, right, resolution)
        out[str(lid)] = [[round(x, 4), round(y, 4)] for x, y in line]
    return out, from_waypoints, inverted


def route_polyline(centerlines, lanelet_ids, goal_xy):
    """The route as one polyline, as the vehicle's planner builds it: the
    preferred lanelets' centerlines in route order (repeated joints dropped),
    cut at the point nearest the goal. Returns (polyline, missing ids)"""
    poly, missing = [], []
    for lid in lanelet_ids:
        line = centerlines.get(str(lid))
        if line is None:
            missing.append(lid)
            continue
        for p in line:
            if not poly or math.dist(poly[-1], p) > 1e-3:
                poly.append(tuple(p))
    if len(poly) > 1 and goal_xy is not None:
        nearest = min(range(len(poly)), key=lambda i: math.dist(poly[i], goal_xy))
        poly = poly[:nearest + 1]
    return poly, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help="lanelet2_map.osm, or the map directory holding it")
    ap.add_argument("out", help="output .json")
    ap.add_argument("--resolution", type=float, default=5.0,
                    help="centerline resolution [m], the map loader's center_line_resolution")
    a = ap.parse_args()
    osm = os.path.join(a.map, "lanelet2_map.osm") if os.path.isdir(a.map) else a.map
    centerlines, from_waypoints, inverted = convert(osm, a.resolution)
    with open(a.out, "w") as f:
        json.dump({"source": os.path.abspath(osm), "resolution_m": a.resolution, "lanelets": centerlines},
                  f, separators=(",", ":"))
    print(f"{len(centerlines)} lanelets ({from_waypoints} from waypoints, {inverted} with the right bound "
          f"inverted) -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
