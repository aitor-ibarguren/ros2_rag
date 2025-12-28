import os
import unittest

import launch
import launch_testing
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import LifecycleNode

from ros2_rag_msgs.srv import LoadCsvData, LoadPdfData, SaveIndex


@pytest.mark.launch_test()
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
class TestDataLoadingLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self._node = rclpy.create_node('test_data_loading_launch')
        self._node.get_logger().info("Node 'test_data_loading_launch' init")

    def tearDown(self):
        self._node.destroy_node()

    def test_load_csv_data(self):
        # Create load CSV data client
        load_csv_data_client = self._node.create_client(
            LoadCsvData,
            'ros2_rag/load_csv_data'
        )

        # Wait until node is ready
        while not load_csv_data_client.wait_for_service(timeout_sec=1.0):
            print("Waiting for 'load_csv_data' service...")

        # Fill request
        req = LoadCsvData.Request()
        req.file_path = (os.path.join(
            get_package_share_directory('ros2_rag'),
            'test', 'data', 'shop_data.csv'
        ))
        req.column_id = 'shop data'
        req.chunking = False

        future = load_csv_data_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        # Check result
        self.assertTrue(future.result().success)

    def test_load_pdf_data(self):
        # Create load PDF data client
        load_pdf_data_client = self._node.create_client(
            LoadPdfData,
            'ros2_rag/load_pdf_data'
        )

        # Wait until node is ready
        while not load_pdf_data_client.wait_for_service(timeout_sec=1.0):
            print("Waiting for 'load_pdf_data' service...")

        # Fill request
        req = LoadPdfData.Request()
        req.folder_path = (os.path.join(
            get_package_share_directory('ros2_rag'),
            'test', 'data'
        ))
        req.chunk_size = 256
        req.chunk_overlap = 25

        future = load_pdf_data_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        # Check result
        self.assertTrue(future.result().success)

    def test_save_index(self):
        # Create load CSV data client
        save_index_client = self._node.create_client(
            SaveIndex,
            'ros2_rag/save_index'
        )

        # Wait until node is ready
        while not save_index_client.wait_for_service(timeout_sec=1.0):
            print("Waiting for 'save_index' service...")

        # Fill request
        req = SaveIndex.Request()
        req.folder_path = ('/tmp/knowledge_base')

        future = save_index_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)

        # Check result
        self.assertTrue(future.result().success)
