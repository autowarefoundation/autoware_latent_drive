#!/usr/bin/env bash
# Run the planner on a rosbag: the latentdrive_ros node, the bag player and
# RViz in one go (ros2/latentdrive_ros/launch/replay.launch.py).
#
#   run_replay.sh <bag_dir> <engine> [launch_arg:=value ...]
#
#   run_replay.sh /bags/1551 models/engine/camfront3-rptw-lane2-e1-mixed.engine
#   run_replay.sh /bags/1551 E.engine start_offset:=60 rate:=0.5
#   run_replay.sh /bags/0930 E.engine route_source:=file route:=/bags/0930/route.npy
#
# <bag_dir> must be playable (metadata.yaml): run prepare_bag.sh on the
# vehicle's .mcap directory once. The package must be built (colcon, see
# ros2/latentdrive_ros/README.md); INSTALL points at its install space,
# default <repo>/ros2/install.
set -eo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BAG=${1:?usage: run_replay.sh <bag_dir> <engine> [launch_arg:=value ...]}
ENGINE=${2:?usage: run_replay.sh <bag_dir> <engine> [launch_arg:=value ...]}
shift 2
INSTALL=${INSTALL:-$REPO/ros2/install}

[ -f "$BAG/metadata.yaml" ] || { echo "$BAG has no metadata.yaml: run samples/rosbag/prepare_bag.sh on it first" >&2; exit 1; }
[ -f "$ENGINE" ] || { echo "no engine at $ENGINE (latentdrive-trt --mode=build-engine)" >&2; exit 1; }
[ -f "$INSTALL/setup.bash" ] || { echo "no latentdrive_ros install at $INSTALL (colcon build, or set INSTALL)" >&2; exit 1; }
[ -n "${ROS_DISTRO:-}" ] || { echo "source /opt/ros/<distro>/setup.bash first" >&2; exit 1; }
# shellcheck disable=SC1091
source "$INSTALL/setup.bash"
exec ros2 launch latentdrive_ros replay.launch.py bag:="$(readlink -f "$BAG")" engine:="$(readlink -f "$ENGINE")" "$@"
