"""Open-loop replay: start the planner node, play the rosbag, show the result.

    ros2 launch latentdrive_ros replay.launch.py \
        bag:=/path/to/bag_dir engine:=/path/to.engine

The engine decides how many cameras are read (the first 1, 2 or 3 of
camera0/camera1/camera2, default CAM_FRONT, FRONT_LEFT, FRONT_RIGHT of the
prd_jt vehicles = camera0, camera3, camera4) and whether the status carries one
subgoal or six route points. Where the route comes from (route_source):

- mission (the default when lanelet_map:= is given): the bag's mission-planning
  route (LaneletRoute) on the lanelet centerlines of lanelet_map:=map.json,
  which scripts/convert_map.py writes from the map's lanelet2_map.osm. Works
  with any Autoware bag that recorded the route;
- recorded (the default otherwise): the route points the vehicle's planner fed
  the model, recorded in the bags of LatentDrive runs;
- file: built from route:=route.npy.

The node publishes /latentdrive_node/viz (camera + BEV image, also as
/viz/compressed), /latentdrive_node/plan (base_link) and, in the map frame,
/latentdrive_node/plan_map and /latentdrive_node/route_points. viewer:=rviz
(default) opens RViz on all of them next to the plan the vehicle recorded;
viewer:=rqt opens rqt_image_view on the image only; viewer:=none is headless.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable,
                            TimerAction)
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

RECORDED_ROUTE_POINTS = "/tensorrt_e2e/debug/latentdrive/route_points"
RECORDED_PLAN = "/tensorrt_e2e/debug/latentdrive/plan"
ROUTE = "/planning/mission_planning/route"


def route_source(context):
    source = LaunchConfiguration("route_source").perform(context)
    if source == "auto":
        return "mission" if LaunchConfiguration("lanelet_map").perform(context) else "recorded"
    return source


def bag_nodes(context):
    """The planner node and the latched-topic publisher, with their parameters
    resolved here: a list of substitutions would reach the node as one
    concatenated string, and a float given as 10 as an integer -- both are
    refused by the declared parameter types"""
    value = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    cameras = [value(f"camera{i}") for i in range(3)]
    source = route_source(context)
    planner = Node(
        package="latentdrive_ros",
        executable="latentdrive_node",
        name="latentdrive_node",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "engine_file": value("engine"),
            "calibration_file": value("calibration"),
            "scene_match": value("scene_match"),
            "route_source": source,
            "route_file": value("route"),
            "lanelet_map": value("lanelet_map"),
            "route_topic": ROUTE,
            "motion_source": value("motion_source"),
            "camera_topics": [c for c in cameras if c],
            "view_tolerance_s": float(value("view_tolerance_s")),
            "missing_view_fill": value("missing_view_fill"),
            "max_rate_hz": float(value("max_rate_hz")),
            "recorded_route_points_topic": RECORDED_ROUTE_POINTS,
        }],
    )
    # /tf_static and the route are recorded once, when they are set, so a bag
    # started at start_offset never plays them; this publishes them, latched,
    # from the bag itself
    latched = Node(
        package="latentdrive_ros", executable="publish_latched.py", name="latched_from_bag",
        arguments=[value("bag"), "--start-offset", value("start_offset"),
                   "--route-topic", ROUTE if source == "mission" else ""],
        output="log",
    )
    return [planner, latched]


def generate_launch_description():
    share = get_package_share_directory("latentdrive_ros")
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
        DeclareLaunchArgument("calibration", default_value="",
                              description="calibration.json for the overlay; empty = from the bag's "
                                          "camera_info and /tf_static"),
        DeclareLaunchArgument("scene_match", default_value="prd_jt",
                              description="calibration.json scene entry to use"),
        DeclareLaunchArgument("route_source", default_value="auto",
                              description="mission (the bag's LaneletRoute + lanelet_map) | recorded (the "
                                          "vehicle's route points in the bag) | file | auto: mission when "
                                          "lanelet_map is given, else recorded"),
        DeclareLaunchArgument("lanelet_map", default_value="",
                              description="lanelet centerlines for route_source:=mission "
                                          "(scripts/convert_map.py <lanelet2_map.osm> map.json)"),
        DeclareLaunchArgument("route", default_value="", description="route.npy for route_source:=file"),
        DeclareLaunchArgument("motion_source", default_value="pose_difference",
                              description="pose_difference (training convention) | twist"),
        DeclareLaunchArgument("camera0", default_value="/sensing/camera/camera0/image_raw/compressed",
                              description="view 0: CAM_FRONT (anchors the history)"),
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
        DeclareLaunchArgument("rate", default_value="1.0", description="bag play rate"),
        DeclareLaunchArgument("start_offset", default_value="0.0",
                              description="start the bag this many seconds in"),
        DeclareLaunchArgument("play_delay", default_value="8.0",
                              description="seconds between node start and bag play (engine load)"),
        DeclareLaunchArgument("viewer", default_value="rviz", description="rviz | rqt | none"),
        OpaqueFunction(function=bag_nodes),
        # The node needs a few seconds to deserialize the engine and warm it
        # up; frames played before that are lost and the first plans would
        # wait for a full 1.5 s history anyway. /tf_static is played latched
        # (qos_override.yaml) so the overlay finds the camera extrinsics, and
        # so is a route recorded later in the bag (a reroute)
        TimerAction(period=arg("play_delay"), actions=[
            ExecuteProcess(
                cmd=["ros2", "bag", "play", arg("bag"), "--clock", "--rate", arg("rate"),
                     "--start-offset", arg("start_offset"),
                     "--qos-profile-overrides-path", qos_override,
                     "--topics", *cameras,
                     "/sensing/camera/camera0/camera_info",
                     "/localization/kinematic_state", "/localization/acceleration",
                     "/tf", "/tf_static", ROUTE, RECORDED_ROUTE_POINTS, RECORDED_PLAN],
                output="screen",
            ),
        ]),
        Node(
            package="rviz2", executable="rviz2", name="rviz2",
            arguments=["-d", os.path.join(share, "rviz", "replay.rviz")],
            parameters=[{"use_sim_time": True}],
            condition=LaunchConfigurationEquals("viewer", "rviz"),
        ),
        Node(
            package="rqt_image_view", executable="rqt_image_view", name="viz_window",
            arguments=["/latentdrive_node/viz"],
            condition=LaunchConfigurationEquals("viewer", "rqt"),
        ),
    ])
