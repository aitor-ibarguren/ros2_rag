import time
import unittest

import launch
import launch_testing
import pytest
import rclpy
from launch_ros.actions import LifecycleNode
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState


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
        self._node = rclpy.create_node('test_lifecycle_launch')
        self._node.get_logger().info("Node 'test_lifecycle_launch' init")

    def tearDown(self):
        self._node.destroy_node()

    def change_state(self, client, transition) -> bool:
        req = ChangeState.Request()
        req.transition.id = transition
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        return future.result().success

    def test_publishes_pose(self):
        # Create lifecycle client
        client = self._node.create_client(ChangeState, 'ros2_rag/change_state')

        # Wait until node is ready
        while not client.wait_for_service(timeout_sec=1.0):
            print("Waiting for lifecycle node service...")

        # Configure
        self.assertTrue(self.change_state(
            client, Transition.TRANSITION_CONFIGURE)
        )

        time.sleep(5.0)

        # Activate
        self.assertTrue(self.change_state(
            client, Transition.TRANSITION_ACTIVATE)
        )
