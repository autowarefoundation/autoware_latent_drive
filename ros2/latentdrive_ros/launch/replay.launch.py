"""Open-loop replay: start the planner node and play the rosbag together.

    ros2 launch latentdrive_ros replay.launch.py \
        bag:=/path/to/bag_dir engine:=/path/to.engine \
        calibration:=/path/to/calibration.json route:=/path/to/route.npy

The engine file decides whether one or two cameras are read: a two-view
engine also subscribes to `far_camera` (default camera2, the telephoto), which
the bag is played with either way. `view_tolerance_s` bounds the wide/telephoto
stamp mismatch per history slot.

The node publishes /latentdrive_node/viz (camera+BEV image) and
/latentdrive_node/plan (nav_msgs/Path). View the image with:
    ros2 run rqt_image_view rqt_image_view /latentdrive_node/viz
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
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
        DeclareLaunchArgument("camera", default_value="/sensing/camera/camera1/image_raw/compressed",
                              description="wide front camera topic (view 0)"),
        DeclareLaunchArgument("far_camera", default_value="/sensing/camera/camera2/image_raw/compressed",
                              description="telephoto camera topic (view 1, two-view engines only)"),
        DeclareLaunchArgument("view_tolerance_s", default_value="0.06",
                              description="max |telephoto stamp - wide stamp| per history slot"),
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
                "camera_topic": LaunchConfiguration("camera"),
                "far_camera_topic": LaunchConfiguration("far_camera"),
                "view_tolerance_s": LaunchConfiguration("view_tolerance_s"),
            }],
        ),
        # Delay the bag so the node has warmed up (engine load + 5 warmup
        # inferences) and rqt has subscribed before playback starts -- otherwise
        # the first frames are missed and the window stays blank.
        TimerAction(period=5.0, actions=[
            ExecuteProcess(
                cmd=["ros2", "bag", "play", bag, "--rate", rate,
                     "--topics",
                     LaunchConfiguration("camera"),
                     LaunchConfiguration("far_camera"),
                     "/localization/kinematic_state",
                     "/localization/acceleration"],
                output="screen",
            ),
        ]),
        # live window (needs X11 forwarding + `xhost +local:docker` on the host)
        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="viz_window",
            arguments=["/latentdrive_node/viz"],
            condition=IfCondition(LaunchConfiguration("display")),
        ),
    ])
