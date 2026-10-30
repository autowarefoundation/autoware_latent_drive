"""Open-loop replay: start the planner node, latch the bag's /tf_static, play the bag.

    ros2 launch autoware_latent_drive replay.launch.py \
        bag:=/path/to/bag_dir engine:=/path/to.engine [out_video:=/path/to.mp4]

The engine decides how many cameras are read (the first 1, 2 or 3 of
camera0/camera1/camera2, default CAM_FRONT, FRONT_LEFT, FRONT_RIGHT of the
prd_jt vehicles = camera0, camera3, camera4) and whether the status carries one
subgoal or six route points. The route points are the ones the vehicle's
planner fed its model, recorded in the bag on
/tensorrt_e2e/debug/latentdrive/route_points (LatentDrive runs).

The node publishes /latentdrive_node/plan (base_link) and shows the plan
projected on the front camera in an OpenCV window (Esc or q closes it;
show_window:=false for a headless run); out_video:= also writes it to an mp4.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable,
                            TimerAction)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROUTE_POINTS = "/tensorrt_e2e/debug/latentdrive/route_points"


def bag_nodes(context):
    """The planner node and the /tf_static publisher, with their parameters
    resolved here: a list of substitutions would reach the node as one
    concatenated string, and a float given as 10 as an integer -- both are
    refused by the declared parameter types"""
    value = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    cameras = [value(f"camera{i}") for i in range(3)]
    planner = Node(
        package="autoware_latent_drive",
        executable="latentdrive_node",
        name="latentdrive_node",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "engine_file": value("engine"),
            "camera_topics": [c for c in cameras if c],
            "view_tolerance_s": float(value("view_tolerance_s")),
            "missing_view_fill": value("missing_view_fill"),
            "max_rate_hz": float(value("max_rate_hz")),
            "motion_source": value("motion_source"),
            "route_points_topic": ROUTE_POINTS,
            "show_window": value("show_window") == "true",
            "out_video": value("out_video"),
        }],
    )
    # /tf_static is recorded once, at the bag start, so a bag started at
    # start_offset never plays it; this publishes it, latched, from the bag
    tf_static = Node(
        package="autoware_latent_drive", executable="publish_tf_static.py", name="tf_static_from_bag",
        arguments=[value("bag"), "--start-offset", value("start_offset")],
        output="log",
    )
    return [planner, tf_static]


def generate_launch_description():
    share = get_package_share_directory("autoware_latent_drive")
    arg = LaunchConfiguration
    cameras = [arg("camera0"), arg("camera1"), arg("camera2")]
    qos_override = os.path.join(share, "config", "qos_override.yaml")
    # Fast DDS (humble's default RMW) sends a ~1 MB camera frame as fragmented
    # UDP once it outgrows the default shared-memory segment, and with the
    # usual 208 KB socket buffer whole stretches of frames are lost. This
    # profile keeps them on shared memory; a profile set by the caller wins,
    # and other RMWs ignore the variable
    fastdds = []
    if "FASTRTPS_DEFAULT_PROFILES_FILE" not in os.environ:
        fastdds = [SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE",
                                          os.path.join(share, "config", "fastdds_replay.xml"))]
    return LaunchDescription(fastdds + [
        DeclareLaunchArgument("bag", description="rosbag directory to replay (with metadata.yaml)"),
        DeclareLaunchArgument("engine", description="TensorRT engine file"),
        DeclareLaunchArgument("camera0", default_value="/sensing/camera/camera0/image_raw/compressed",
                              description="view 0: CAM_FRONT (anchors the history, drawn on)"),
        DeclareLaunchArgument("camera1", default_value="/sensing/camera/camera3/image_raw/compressed",
                              description="view 1: CAM_FRONT_LEFT (multi-view engines only)"),
        DeclareLaunchArgument("camera2", default_value="/sensing/camera/camera4/image_raw/compressed",
                              description="view 2: CAM_FRONT_RIGHT (three-view engines only)"),
        DeclareLaunchArgument("view_tolerance_s", default_value="0.05",
                              description="max |side stamp - front stamp| per history slot"),
        DeclareLaunchArgument("missing_view_fill", default_value="skip",
                              description="skip | zeros (the blank frame the camfront3 training used)"),
        DeclareLaunchArgument("max_rate_hz", default_value="10.0",
                              description="plans per second at most, paced on the view-0 stamps"),
        DeclareLaunchArgument("motion_source", default_value="pose_difference",
                              description="pose_difference (training convention) | twist"),
        DeclareLaunchArgument("rate", default_value="1.0", description="bag play rate"),
        DeclareLaunchArgument("start_offset", default_value="0.0",
                              description="start the bag this many seconds in"),
        DeclareLaunchArgument("play_delay", default_value="8.0",
                              description="seconds between node start and bag play (engine load)"),
        DeclareLaunchArgument("show_window", default_value="true",
                              description="false: no OpenCV window (headless; use out_video)"),
        DeclareLaunchArgument("out_video", default_value="",
                              description="also write the camera overlay to this mp4"),
        OpaqueFunction(function=bag_nodes),
        # The node needs a few seconds to deserialize the engine and warm it
        # up; frames played before that are lost and the first plans would
        # wait for a full 1.5 s history anyway. /tf_static is played latched
        # (qos_override.yaml) so the overlay finds the camera extrinsics
        TimerAction(period=arg("play_delay"), actions=[
            ExecuteProcess(
                cmd=["ros2", "bag", "play", arg("bag"), "--clock", "--rate", arg("rate"),
                     "--start-offset", arg("start_offset"),
                     "--qos-profile-overrides-path", qos_override,
                     "--topics", *cameras,
                     "/sensing/camera/camera0/camera_info",
                     "/localization/kinematic_state", "/localization/acceleration",
                     "/tf_static", ROUTE_POINTS],
                output="screen",
            ),
        ]),
    ])
