import os
import unittest

import launch
import launch_testing
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import LifecycleNode
from ros2_rag_msgs.srv import Query, RAGQuery


@pytest.mark.launch_test
def generate_test_description():

    # Load yaml
    yaml_path = os.path.join(
        get_package_share_directory('ros2_rag'),
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

    def test_query(self):
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

        future = query_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        # Check result
        self.assertTrue(future.result().success and len(
            future.result().completion) > 0)

    def test_rag_query(self):
        # Create load CSV data client
        rag_query_client = self._node.create_client(
            RAGQuery,
            'ros2_rag/rag_query'
        )

        # Wait until node is ready
        while not rag_query_client.wait_for_service(timeout_sec=1.0):
            print("Waiting for 'rag_query' service...")

        # Fill request
        req = RAGQuery.Request()
        req.query = 'Can I buy a Gibson brand guitar in the shop?'
        req.query_template = ('You are a helpful AI assistant. Use the ' +
                              'context below to answer the user question.\n' +
                              '--- CONTEXT START ---\n' +
                              '%context%\n' +
                              '--- CONTEXT END ---' +
                              '--- QUESTION ---\n' +
                              '%query%\n' +
                              '--- ANSWER ---\n'
                              )

        future = rag_query_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        # Check result
        self.assertTrue(future.result().success and len(
            future.result().completion) > 0)
