"""Open-loop replay: start the planner node and play the rosbag together.

    ros2 launch latentdrive_ros replay.launch.py \
        bag:=/path/to/bag_dir engine:=/path/to.engine \
        calibration:=/path/to/calibration.json route:=/path/to/route.npy

The node publishes /latentdrive_node/viz (camera+BEV image) and
/latentdrive_node/plan (nav_msgs/Path). View the image with:
    ros2 run rqt_image_view rqt_image_view /latentdrive_node/viz
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bag = LaunchConfiguration("bag")
    rate = LaunchConfiguration("rate")
    return LaunchDescription([
        DeclareLaunchArgument("bag", description="rosbag directory to replay"),
        DeclareLaunchArgument("engine", description="TensorRT engine file"),
        DeclareLaunchArgument("calibration", description="calibration.json"),
        DeclareLaunchArgument("route", description="route.npy from export_route.py"),
        DeclareLaunchArgument("rate", default_value="1.0", description="bag play rate"),
        DeclareLaunchArgument("display", default_value="true",
                              description="open an rqt_image_view window on /viz"),
        Node(
            package="latentdrive_ros",
            executable="latentdrive_node",
            name="latentdrive_node",
            output="screen",
            parameters=[{
                "engine_file": LaunchConfiguration("engine"),
                "calibration_file": LaunchConfiguration("calibration"),
                "route_file": LaunchConfiguration("route"),
            }],
        ),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", bag, "--rate", rate,
                 "--topics",
                 "/sensing/camera/camera1/image_raw/compressed",
                 "/localization/kinematic_state",
                 "/localization/acceleration"],
            output="screen",
        ),
        # live window (needs X11 forwarding + `xhost +local:docker` on the host)
        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="viz_window",
            arguments=["/latentdrive_node/viz"],
            condition=IfCondition(LaunchConfiguration("display")),
        ),
    ])
