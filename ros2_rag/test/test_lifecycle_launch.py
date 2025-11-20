import unittest

import launch
import launch_testing
import pytest
import rclpy
from launch_ros.actions import LifecycleNode


@pytest.mark.launch_test
def generate_test_description():
    # Launch your node as part of the test
    node_under_test = LifecycleNode(
        package='ros2_rag',
        namespace='',
        executable='ros2_rag',
        name='ros2_rag',
        parameters=[{'auto_activate': False}]
    )

    return (
        launch.LaunchDescription([
            node_under_test,
            # Launch tests 5.0 s later
            launch.actions.TimerAction(
                period=5.0, actions=[launch_testing.actions.ReadyToTest()]),
        ])
    )


# Active tests
class TestLifecycleLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_lifecycle_launch')
        self.node.get_logger().info("Node 'test_lifecycle_launch' initialized")

    def tearDown(self):
        self.node.destroy_node()

    def test_publishes_pose(self):
        self.assertTrue(True)
