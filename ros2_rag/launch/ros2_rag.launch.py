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

    # ROS2 RAG
    ros2_rag_node = LifecycleNode(
        package='ros2_rag',
        namespace='',
        executable='ros2_rag',
        name='ros2_rag',
        parameters=[{'auto_activate': LaunchConfiguration('auto_activate')}]
    )

    ld.add_action(ros2_rag_node)

    return ld
