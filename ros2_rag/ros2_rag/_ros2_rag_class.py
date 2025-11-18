from typing import Dict

from llm_wrappers.flan_t5_wrapper.flan_t5_wrapper import (FlanT5Type,
                                                          FlanT5Wrapper)
from rclpy.lifecycle import LifecycleNode
from ros2_rag_msgs.srv import Query


class ROS2RAGClass:

    def __init__(self, node: LifecycleNode):
        # Copy vars
        self._node = node
        self._logger = self._node.get_logger()

        # Init vars
        self._generator = None

        self._load_data_srv = None

    def __del__(self):
        self._logger.info('Destroying ROS2RAGClass instance...')

    def configure(self, params: Dict[str, str]) -> bool:
        # Instantiate generator
        if params['generator_type'] == 'flan_t5':
            self._generator = FlanT5Wrapper(FlanT5Type.SMALL)
            self._logger.info('LLM model: Flan T5 - SMALL')

        # Load from pretrained
        if params['generator_loading'] == 'pretrained':
            self._generator.load_pretrained_model()
            self._logger.info('Flan T5 generator ready')

        return True

    def activate(self) -> bool:
        # Initialize services
        self._load_data_srv = self._node.create_service(
            Query, self._node.get_name()+'/query', self.query_callback)
        self._logger.info('Query service ready ✅')

        return True
    
    def deactivate(self) -> bool:
        # Shutdown services
        self._node.destroy_service(self._load_data_srv)
        self._logger.info('Query service shutdown ✔️')

        return True

    def query_callback(self, request, response):
        self._logger.info('QUERY request received...')

        # Get completion
        completion = ''
        res, completion = self._generator.generate(request.query)

        if res:
            response.completion = completion
            response.success = True
            response.error_code = 0
            response.error_string = ''

            msg = 'Completion generated and sent ✨'
        else:
            response.completion = ''
            response.success = False
            response.error_code = -1
            response.error_string = 'Error generating completion'

            msg = '❌' + response.error_string

        # Finish
        self._logger.info(f"{msg}")

        return response
