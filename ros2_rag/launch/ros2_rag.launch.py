import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    # Create launch description
    ld = LaunchDescription()

    # Declare launch arguments
    auto_activate_arg = DeclareLaunchArgument(
        'auto_activate',
        default_value='False',
        description='Auto activate ROS2 RAG lifecycle node'
    )

    ld.add_action(auto_activate_arg)

    # Load yaml
    yaml_path = os.path.join(
        get_package_share_directory('ros2_rag'),
        'config',
        'ros2_rag_params.yml'
    )

    # ROS2 RAG
    ros2_rag_node = LifecycleNode(
        package='ros2_rag',
        namespace='',
        executable='ros2_rag',
        name='ros2_rag',
        parameters=[
            yaml_path,
            {'auto_activate': LaunchConfiguration('auto_activate')}
        ]
    )

    ld.add_action(ros2_rag_node)

    return ld
