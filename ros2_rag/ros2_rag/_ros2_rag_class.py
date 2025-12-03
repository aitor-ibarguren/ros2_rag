import os
from typing import Dict, Tuple

from llm_wrappers.bart_wrapper.bart_wrapper import BARTType, BARTWrapper
from llm_wrappers.faiss_wrapper.faiss_wrapper import FAISSWrapper
from llm_wrappers.flan_t5_wrapper.flan_t5_wrapper import (FlanT5Type,
                                                          FlanT5Wrapper)
from llm_wrappers.gpt2_wrapper.gpt2_wrapper import GPT2Type, GPT2Wrapper
from rclpy.lifecycle import LifecycleNode

from ros2_rag_msgs.srv import LoadCsvData, Query, RAGQuery, SaveIndex


class ROS2RAGClass:

    def __init__(self, node: LifecycleNode):
        # Copy vars
        self._node = node
        self._logger = self._node.get_logger()

        # Generator family/versions
        self._ALLOWED_GENERATORS = {
            "flan_t5": ["small", "base", "large", "xl", "xxl"],
            "bart": ["base", "large"],
            "gpt2": ["base", "medium", "large", "xl"],
        }

        # Init vars
        self._generator = None
        self._retriever = None

        self._load_data_srv = None

    def __del__(self):
        self._logger.info('Destroying ROS2RAGClass instance...')

    def _get_params(self) -> Tuple[bool, Dict[str, str]]:
        params = {}

        # Declare parameters
        self._node.declare_parameter('generator_family', 'flan_t5')
        self._node.declare_parameter('generator_version', 'small')
        self._node.declare_parameter('generator_loading', 'pretrained')
        self._node.declare_parameter('knowledge_base_path', '~/knowledge_base')

        # Get parameter values
        params['generator_family'] = self._node.get_parameter(
            'generator_family').value
        params['generator_version'] = self._node.get_parameter(
            'generator_version').value
        params['generator_loading'] = self._node.get_parameter(
            'generator_loading').value
        params['knowledge_base_path'] = self._node.get_parameter(
            'knowledge_base_path').value

        return True, params

    def _check_params(self, params: Dict[str, str]) -> Tuple[bool, str]:
        # Check generator family
        if params['generator_family'] not in self._ALLOWED_GENERATORS:
            return False, ("Generator family '" + params['generator_family'] +
                           "' unknown")

        # Check generator version
        if params['generator_version'] not in self._ALLOWED_GENERATORS[
                params['generator_family']]:
            return False, ("Generator version '" + params['generator_version']
                           + "' unknown for family '" +
                           params['generator_family'] + "'")

        return True, ''

    def _init_retriever(self, knowledge_base_path: str) -> Tuple[bool, str]:
        # Add slash at the end of path if required
        if not knowledge_base_path.endswith('/'):
            knowledge_base_path += '/'

        # Check if knowledge base folder exists
        if (not os.path.exists(knowledge_base_path) or
                not os.path.isdir(knowledge_base_path)):
            # Create folder
            self._node.get_logger().info(
                f"Creating new knowledge base in folder "
                f"'{knowledge_base_path}'..."
            )
            os.makedirs(knowledge_base_path, exist_ok=True)

        # Instantiate FAISS wrapper
        self._retriever = FAISSWrapper()

        # Check if knowledge base empty
        if (not os.path.exists(knowledge_base_path + 'knowledge_base.faiss') or
                not os.path.isfile(knowledge_base_path +
                                   'knowledge_base.faiss')):
            # Init new index
            self._node.get_logger().info(
                f"Knowledge base in folder "
                f"'{knowledge_base_path}' empty, initializing new index..."
            )
            self._retriever.init_new_index()
        else:
            # Load index
            self._node.get_logger().info(
                f"Loading ⏳ knowledge base from folder "
                f"'{knowledge_base_path}'..."
            )

            if not self._retriever.load_stored_index(
                    knowledge_base_path, 'knowledge_base'):

                error_msg = 'Error loading stored index'
                self._node.get_logger().error(f'❌ {error_msg}')

                return False, error_msg

            # Feedback
            _, size = self._retriever.get_index_size()
            self._node.get_logger().info(f'Loaded index: {size} items')

        return True, ''

    def configure(self) -> bool:
        # Get params
        res, params = self._get_params()

        if not res:
            self._node.get_logger().error('❌ Error retrieving parameters')
            return False

        # Check params
        res, error_msg = self._check_params(params)

        if not res:
            self._node.get_logger().error(
                f'❌ Error in parameters: {error_msg}')
            return False

        # Init retriever & knowledge base
        res, error_msg = self._init_retriever(params['knowledge_base_path'])

        if not res:
            self._node.get_logger().error(
                f'❌ Error initializing retriever: {error_msg}')
            return False

        self._logger.info('Retriever successfully initialized 🎯')

        # Instantiate generator
        version_idx = self._ALLOWED_GENERATORS[
            params['generator_family']].index(
                params['generator_version'])

        if params['generator_family'] == 'bart':
            version = list(BARTType)[version_idx]
            self._generator = BARTWrapper(list(BARTType)[version_idx])
            self._logger.info(f'LLM model: BART - {version.name}')
        elif params['generator_family'] == 'flan_t5':
            version = list(FlanT5Type)[version_idx]
            self._generator = FlanT5Wrapper(list(FlanT5Type)[version_idx])
            self._logger.info(f'LLM model: Flan T5 - {version.name}')
        elif params['generator_family'] == 'gpt2':
            version = list(GPT2Type)[version_idx]
            self._generator = GPT2Wrapper(list(GPT2Type)[version_idx])
            self._logger.info(f'LLM model: GPT2 - {version.name}')

        # Load generator
        if params['generator_loading'] == 'pretrained':
            self._logger.info('Loading ⏳ pretrained model...')
            self._generator.load_pretrained_model()

        self._logger.info('Model successfully loaded 🎯')

        return True

    def activate(self) -> bool:
        # Initialize services
        self._load_csv_data_srv = self._node.create_service(
            LoadCsvData, self._node.get_name()+'/load_csv_data',
            self.load_csv_data_callback)
        self._logger.info('Load CSV data service ready ✅')

        self._save_index_srv = self._node.create_service(
            SaveIndex, self._node.get_name()+'/save_index',
            self.save_index_callback)
        self._logger.info('Save index service ready ✅')

        self._query_srv = self._node.create_service(
            Query, self._node.get_name()+'/query', self.query_callback)
        self._logger.info('Query service ready ✅')

        self._rag_query_srv = self._node.create_service(
            RAGQuery, self._node.get_name()+'/rag_query',
            self.rag_query_callback)
        self._logger.info('RAG query service ready ✅')

        return True

    def deactivate(self) -> bool:
        # Shutdown services
        self._node.destroy_service(self._load_data_srv)
        self._logger.info('Query service shutdown ✔️')

        return True

    def load_csv_data_callback(self, request, response):
        self._logger.info('LOAD CSV DATA request received...')

        # Get initial index size
        _, initial_size = self._retriever.get_index_size()

        # Load CSV data
        self._logger.info(f"Loading '{request.file_path}'...")

        if not self._retriever.add_from_csv(
                request.file_path,
                request.column_id,
                chunking=request.chunking,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap):
            response.success = False
            response.error_code = -1
            response.error_msg = 'Error loading CSV file'

            self._logger.error('❌ ' + response.error_msg)
        else:
            # Get new index size
            _, new_size = self._retriever.get_index_size()

            self._logger.info(
                f"Added {new_size - initial_size} new items to index 📚 "
                f"(total index size {new_size})")

            response.success = True
            response.error_code = 0
            response.error_msg = ''

        return response

    def save_index_callback(self, request, response):
        self._logger.info('SAVE INDEX request received...')

        # Save index
        if not self._retriever.save_index(
                request.folder_path,
                'knowledge_base'):
            response.success = False
            response.error_code = -1
            response.error_msg = 'Error saving index'

            self._logger.error('❌ ' + response.error_msg)
        else:
            response.success = True
            response.error_code = 0
            response.error_msg = ''

            self._logger.info(f"Index saved on '{request.folder_path}' 📥")

        return response

    def query_callback(self, request, response):
        self._logger.info('QUERY request received...')

        # Get completion
        completion = ''
        res, completion = self._generator.generate(request.query)

        if res:
            response.completion = completion
            response.success = True
            response.error_code = 0
            response.error_msg = ''

            self._logger.info('Completion generated and sent ✨')
        else:
            response.completion = ''
            response.success = False
            response.error_code = -1
            response.error_msg = 'Error generating completion'

            self._logger.error('❌ ' + response.error_msg)

        return response

    def _verify_query_template(self, query_template: str) -> Tuple[bool, str]:
        # Check context tag
        if query_template.count('%context%') != 1:
            error_msg = ("The query template must contain a '%context%' tag " +
                         "to insert the information from the knowledge base")

            return False, error_msg
        elif query_template.count('%query%') != 1:
            error_msg = ("The query template must contain a '%query%' tag to" +
                         " insert the provided query")

            return False, error_msg

        return True, ''

    def rag_query_callback(self, request, response):
        self._logger.info('RAG QUERY request received...')

        # Verify template
        res, error_msg = self._verify_query_template(request.query_template)

        if not res:
            response.completion = ''
            response.success = False
            response.error_code = -1
            response.error_msg = error_msg

            self._logger.error('❌ ' + response.error_msg)
            return response

        # Verify that knowledge base contains information
        _, size = self._retriever.get_index_size()

        if size == 0:
            response.completion = ''
            response.success = False
            response.error_code = -2
            response.error_msg = ("Knowledge base empty: Add information " +
                                  "before a RAG query")

            self._logger.error('❌ ' + response.error_msg)
            return response

        # Get the context information from knowledge base
        res, contexts, _ = self._retriever.hybrid_search(
            [request.query], 5, 0.6)

        # Insert context
        augmented_prompt = request.query_template.replace('%context%',
                                                          "\n".join(
                                                              contexts[0]))
        # Insert query
        augmented_prompt = augmented_prompt.replace('%query%', request.query)

        # Get completion
        completion = ''
        res, completion = self._generator.generate(augmented_prompt)

        if res:
            response.completion = completion
            response.success = True
            response.error_code = 0
            response.error_msg = ''

            self._logger.info('Completion generated and sent ✨')
        else:
            response.completion = ''
            response.success = False
            response.error_code = -3
            response.error_msg = 'Error generating completion'

            self._logger.error('❌ ' + response.error_msg)

        return response
