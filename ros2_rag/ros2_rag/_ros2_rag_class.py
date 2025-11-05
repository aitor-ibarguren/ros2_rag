from typing import Tuple

from rclpy.impl.rcutils_logger import RcutilsLogger


class ROS2RAGClass:

    def __init__(self, logger: RcutilsLogger):
        # Copy vars
        self._logger = logger

    def __del__(self):
        self._logger.info("Destroying ROS2RAGClass instance...")

    def query(self, request, response):
        self._logger.info("QUERY request received...")

        # ToDo

        # Finish
        self._logger.info("Completion generated and sent")

        return response
