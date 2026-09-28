"""Carter Nav2 bringup plus a localization-only laser scan.

The stock scan (from the 3D lidar, 0.526 m up) starts 12.6 cm above the
floor, so it contains the terrain patches: a 28 cm ramp, 17 cm rough peaks.
None of that is in the static map, and AMCL drifts trying to match it.

AMCL gets its own scan starting 35 cm above the floor: walls and shelving
stay in, terrain drops out. Obstacle layers keep the original low scan so
they still see pallets. Used for baseline and terrain runs alike.

    ros2 launch bringup/launch/terrain_nav.launch.py params_file:=<yaml>
"""

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

LIDAR_HEIGHT = 0.526          # base_link -> front_3d_lidar, measured with tf2_echo
LOCALIZATION_FLOOR = 0.35     # metres above ground; clears the 0.28 m ramp


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    carter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("carter_navigation"), "launch",
            "carter_navigation.launch.py")),
        launch_arguments={"params_file": params_file}.items(),
    )
    localization_scan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan_localization",
        remappings=[("cloud_in", "/front_3d_lidar/lidar_points"),
                    ("scan", "/scan_localization")],
        parameters=[{
            "use_sim_time": True,
            "target_frame": "front_3d_lidar",
            "transform_tolerance": 0.01,
            "min_height": LOCALIZATION_FLOOR - LIDAR_HEIGHT,
            "max_height": 1.5,
            "angle_min": -math.pi,
            "angle_max": math.pi,
            "angle_increment": math.radians(0.5),
            "scan_time": 0.1,
            "range_min": 0.05,
            "range_max": 100.0,
            "use_inf": True,
        }],
        output="screen",
    )
    return LaunchDescription([
        DeclareLaunchArgument("params_file", description="Nav2 params yaml"),
        carter,
        localization_scan,
    ])
