#!/usr/bin/env bash
# Make a vehicle rosbag (a directory of .mcap files, as the vehicle records it,
# usually without metadata.yaml) playable by `ros2 bag play`, without copying
# or modifying it: a directory of symlinks plus a metadata.yaml that
# `ros2 bag reindex` writes. Run it where ROS 2 is sourced (the container).
#
#   prepare_bag.sh <bag_dir> <out_dir>
#
# On humble the bag's jazzy-format QoS strings cannot be parsed ("yaml-cpp:
# bad conversion"), so they are blanked (= default QoS); the replay launch
# plays /tf_static latched through its own QoS override either way.
set -eo pipefail
SRC=${1:?usage: prepare_bag.sh <bag_dir> <out_dir>}
OUT=${2:?usage: prepare_bag.sh <bag_dir> <out_dir>}
command -v ros2 > /dev/null || { echo "ros2 not found: source /opt/ros/<distro>/setup.bash first" >&2; exit 1; }
shopt -s nullglob
files=("$SRC"/*.mcap)
[ ${#files[@]} -gt 0 ] || { echo "no .mcap files in $SRC" >&2; exit 1; }

mkdir -p "$OUT"
for f in "${files[@]}"; do ln -sfn "$(readlink -f "$f")" "$OUT/"; done
rm -f "$OUT/metadata.yaml"
ros2 bag reindex -s mcap "$OUT" > /dev/null
if [ "${ROS_DISTRO:-}" = humble ]; then
  sed -i 's/offered_qos_profiles: ".*"$/offered_qos_profiles: ""/' "$OUT/metadata.yaml"
fi
ros2 bag info "$OUT" | grep -E "Duration|Messages|camera[0-9]/image_raw/compressed|kinematic_state|latentdrive/route_points" || true
echo "ready: $OUT"
