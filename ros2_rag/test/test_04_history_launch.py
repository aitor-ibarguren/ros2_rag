import os
import unittest

import launch
import launch_testing
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import LifecycleNode

from ros2_rag_msgs.srv import GetHistory, Query


@pytest.mark.launch_test(timeout=30)
def generate_test_description():

    # Load yaml
    yaml_path = os.path.join(
        get_package_share_directory('ros2_rag'),
        'test',
        'config',
        'ros2_rag_params.yml'
    )

    # Launch your node as part of the test
    node_under_test = LifecycleNode(
        package='ros2_rag',
        namespace='',
        executable='ros2_rag',
        name='ros2_rag',
        parameters=[
            yaml_path,
            {'auto_activate': True}
        ]
    )

    return (
        launch.LaunchDescription([
            node_under_test,
            # Launch tests 10.0 s later
            launch.actions.TimerAction(
                period=10.0, actions=[launch_testing.actions.ReadyToTest()]),
        ])
    )


# Active tests
class TestHistoryLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self._node = rclpy.create_node('test_history_launch')
        self._node.get_logger().info("Node 'test_history_launch' init")

    def tearDown(self):
        self._node.destroy_node()

    def query(self, query: str) -> bool:
        # Create load CSV data client
        query_client = self._node.create_client(
            Query,
            'ros2_rag/query'
        )

        # Wait until node is ready
        while not query_client.wait_for_service(timeout_sec=1.0):
            print("Waiting for 'query' service...")

        # Fill request
        req = Query.Request()
        req.query = ('CONTEXT: Pytest popularity and widespread adoption in' +
                     ' the Python community saw a significant rise around ' +
                     'the mid-2010s, particularly following the release of ' +
                     'a key major version. ' +
                     'QUESTION: When did Pytest became popular?')
        req.return_answer_only = True
        req.remove_incomplete_sentences = False

        future = query_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        return future.result().success

    def test_history(self):
        # Create get History client
        get_history_client = self._node.create_client(
            GetHistory,
            'ros2_rag/get_history'
        )

        # Wait until node is ready
        while not get_history_client.wait_for_service(timeout_sec=1.0):
            print("Waiting for 'get_history' service...")

        # Fill request
        req = GetHistory.Request()

        future = get_history_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        # Check result
        self.assertTrue(future.result().success and len(
            future.result().recent_interactions) == 0)
