import os
import re
import threading
from typing import Dict, Tuple

from llm_wrappers.deepseek_wrapper.deepseek_wrapper import (DeepseekType,
                                                            DeepseekWrapper)
from llm_wrappers.faiss_wrapper.faiss_wrapper import FAISSWrapper
from llm_wrappers.qwen_wrapper.qwen_wrapper import QwenType, QwenWrapper
from rclpy.lifecycle import LifecycleNode

from ros2_rag._conversation_history_manager import ConversationHistoryManager
from ros2_rag._rag_verification_manager import RAGVerificationManager
from ros2_rag_msgs.msg import Interaction
from ros2_rag_msgs.srv import (GetHistory, LoadCsvData, LoadPdfData, Query,
                               RAGQuery, SaveIndex)


class ROS2RAGClass:

    def __init__(self, node: LifecycleNode):
        # Copy vars
        self._node = node
        self._logger = self._node.get_logger()

        # Generator family/versions
        self._ALLOWED_GENERATORS = {
            'qwen': ['xtiny', 'tiny', 'small', 'base', 'large', 'xl'],
            'deepseek': ['r1_distill_qwen_tiny', 'r1_distill_qwen_base',
                         'r1_distill_llama_base', 'r1_distill_qwen_large',
                         'r1_distill_qwen_xl', 'r1_distill_llama_xl']
        }

        # Init vars
        self._generator = None
        self._retriever = None

        # Init conversation history manager
        self._conversation_history_manager = None

        # Init RAG verification manager
        self._rag_verification_manager = None

        # Locks
        self._generator_lock = threading.Lock()
        self._summary_lock = threading.Lock()

        # Services
        self._get_history_srv = None
        self._load_data_srv = None
        self._load_pdf_data_srv = None
        self._save_index_srv = None
        self._query_srv = None
        self._rag_query_srv = None

    def __del__(self):
        self._logger.info('Destroying ROS2RAGClass instance...')

    def _get_params(self) -> Tuple[bool, Dict[str, str]]:
        params = {}

        # Declare parameters
        self._node.declare_parameter('generator_family', 'qwen')
        self._node.declare_parameter('generator_version', 'small')
        self._node.declare_parameter('generator_loading', 'pretrained')
        self._node.declare_parameter('knowledge_base_path', '~/knowledge_base')
        self._node.declare_parameter('retriever.top_k', 5)
        self._node.declare_parameter('retriever.alpha', 0.6)
        self._node.declare_parameter('generator.max_new_tokens', 50)
        self._node.declare_parameter('generator.temperature', 0.5)
        self._node.declare_parameter('generator.top_p', 0.9)
        self._node.declare_parameter('history_active', True)
        self._node.declare_parameter('history.recent_interaction_number', 5)
        self._node.declare_parameter('history.evicted_interaction_number', 10)
        self._node.declare_parameter('format.remove_bullets', False)
        self._node.declare_parameter('rag_verification_active', True)
        self._node.declare_parameter('rag_verification.entailment_threshold',
                                     0.75)

        # Get parameter values
        params['generator_family'] = self._node.get_parameter(
            'generator_family').value
        params['generator_version'] = self._node.get_parameter(
            'generator_version').value
        params['generator_loading'] = self._node.get_parameter(
            'generator_loading').value
        params['knowledge_base_path'] = self._node.get_parameter(
            'knowledge_base_path').value
        params['retriever.top_k'] = self._node.get_parameter(
            'retriever.top_k').value
        params['retriever.alpha'] = self._node.get_parameter(
            'retriever.alpha').value
        params['generator.max_new_tokens'] = self._node.get_parameter(
            'generator.max_new_tokens').value
        params['generator.temperature'] = self._node.get_parameter(
            'generator.temperature').value
        params['generator.top_p'] = self._node.get_parameter(
            'generator.top_p').value
        params['history_active'] = self._node.get_parameter(
            'history_active').value
        params['history.recent_interaction_number'] = self._node.get_parameter(
            'history.recent_interaction_number').value
        params[
            'history.evicted_interaction_number'
        ] = self._node.get_parameter(
            'history.evicted_interaction_number').value
        params['format.remove_bullets'] = self._node.get_parameter(
            'format.remove_bullets').value
        params['rag_verification_active'] = self._node.get_parameter(
            'rag_verification_active').value
        params['rag_verification.entailment_threshold'] = (
            self._node.get_parameter(
                'rag_verification.entailment_threshold').value)

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

        # Check retriever top K is int between 1 and 25
        if ((not isinstance(params['retriever.top_k'], (int))) or
                params['retriever.top_k'] < 1 or
                params['retriever.top_k'] > 25):
            return False, ("Retriever - Top K MUST be between 1 and 25")

        # Check retriever alpha is between 0.0 and 1.0
        if ((not isinstance(params['retriever.alpha'], (int, float))) or
                params['retriever.alpha'] < 0.0 or
                params['retriever.alpha'] > 1.0):
            return False, ('Retriever - Alpha MUST be between 0.0 and 1.0')

        # Check generator max new tokens is a positive int
        if (not isinstance(params['generator.max_new_tokens'], int) or
                params['generator.max_new_tokens'] < 1):
            return (False,
                    ('Generator - Max. new tokens MUST be a positive integer'))

        # Check generator temperature is between 0.0 and 1.5
        if (not isinstance(params['generator.temperature'], (int, float)) or
                params['generator.temperature'] < 0.0 or
                params['generator.temperature'] > 1.5):
            return (False,
                    ('Generator - Temperature MUST be between 0.0 and 1.5'))

        # Check generator top P is between 0.0 and 1.0
        if ((not isinstance(params['generator.top_p'], (int, float))) or
                params['generator.top_p'] < 0.0 or
                params['generator.top_p'] > 1.0):
            return False, ('Generator - Top P MUST be between 0.0 and 1.0')

        # Check history active
        if (not isinstance(params['history_active'], (bool))):
            return False, ('History active MUST be true or false')

        # Check history recent interaction number if active
        if params['history_active']:
            if ((not isinstance(params['history.recent_interaction_number'],
                                (int)))
                or params['history.recent_interaction_number'] < 1 or
                    params['history.recent_interaction_number'] > 25):
                return (
                    False,
                    ('Recent interaction number MUST be between 1 and 25'))

        # Check history evicted interaction number if active
        if params['history_active']:
            if ((not isinstance(params['history.evicted_interaction_number'],
                                (int)))
                or params['history.evicted_interaction_number'] < 1 or
                    params['history.evicted_interaction_number'] > 25):
                return (
                    False,
                    ('Evicted interaction number MUST be between 1 and 25'))

        # Check remove bullets
        if (not isinstance(params['format.remove_bullets'], (bool))):
            return False, ('Remove bullets (format) MUST be true or false')

        # Check RAG verification active
        if (not isinstance(params['rag_verification_active'], (bool))):
            return False, ('RAG verification MUST be true or false')

        # Check RAG verification - Entailment threshold
        if (not isinstance(params['rag_verification.entailment_threshold'],
                           (int, float)) or
                params['rag_verification.entailment_threshold'] < 0.0 or
                params['rag_verification.entailment_threshold'] > 1.0):
            return (False,
                    ('RAG verification - Entailment threshold MUST be \
                     between 0.0 and 1.0'))

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

        if params['generator_family'] == 'deepseek':
            version = list(DeepseekType)[version_idx]
            self._generator = DeepseekWrapper(list(DeepseekType)[version_idx])
            self._logger.info(f'LLM model: Deepseek - {version.name}')
        elif params['generator_family'] == 'qwen':
            version = list(QwenType)[version_idx]
            self._generator = QwenWrapper(list(QwenType)[version_idx])
            self._logger.info(f'LLM model: QWEN - {version.name}')

        # Load generator
        if params['generator_loading'] == 'pretrained':
            self._logger.info('Loading ⏳ pretrained model...')
            self._generator.load_pretrained_model()

        self._logger.info('Model successfully loaded 🎯')

        # Store generator family
        self._generator_family = params['generator_family']

        # Print retriever params
        self._logger.info('Retriever parameters 🎛️')
        self._retriever_top_k = params['retriever.top_k']
        self._logger.info(f'► Top K: {self._retriever_top_k}')
        self._retriever_alpha = params['retriever.alpha']
        self._logger.info(f'► Alpha: {self._retriever_alpha}')

        # Print generation params
        self._logger.info('Generation parameters 🎛️')
        self._gen_max_new_tokens = params['generator.max_new_tokens']
        self._logger.info(f'► Max new tokens: {self._gen_max_new_tokens}')
        self._gen_temperature = params['generator.temperature']
        self._logger.info(f'► Temperature: {self._gen_temperature}')
        self._gen_top_p = params['generator.top_p']
        self._logger.info(f'► Top P: {self._gen_top_p}')

        # Print history params if active
        self._history_active = params['history_active']
        if self._history_active:
            self._logger.info('History parameters 🎛️')
            self._recent_interaction_number = params[
                'history.recent_interaction_number']
            self._evicted_interaction_number = params[
                'history.evicted_interaction_number']

            self._logger.info(
                '► Recent interaction number: ' +
                f'{self._recent_interaction_number}')
            self._logger.info(
                '► Evicted interaction number: ' +
                f'{self._evicted_interaction_number}')

            # Init history
            self._conversation_history_manager = ConversationHistoryManager(
                self._recent_interaction_number,
                self._evicted_interaction_number
            )

        # Print format params
        self._logger.info('Format parameters 🎛️')
        self._format_remove_bullets = params['format.remove_bullets']
        self._logger.info(f'► Remove bullets: {self._format_remove_bullets}')

        # Print RAG verification params
        self._rag_verification_active = params['rag_verification_active']
        if self._rag_verification_active:
            self._logger.info('RAG verification parameters 🎛️')
            self._rag_verification_entailment_threshold = params[
                'rag_verification.entailment_threshold']

            self._logger.info(
                '► Entailment threshold: ' +
                f'{self._rag_verification_entailment_threshold}')

            # Init RAG verification manager
            self._rag_verification_manager = RAGVerificationManager()
            self._logger.info(
                'RAG verification successfully initialized 🎯')

        return True

    def activate(self) -> bool:
        # Initialize services
        if self._history_active:
            self._gwet_history_srv = self._node.create_service(
                GetHistory, self._node.get_name()+'/get_history',
                self.get_history_callback)
            self._logger.info('Get history service ready ✅')
        self._load_csv_data_srv = self._node.create_service(
            LoadCsvData, self._node.get_name()+'/load_csv_data',
            self.load_csv_data_callback)
        self._logger.info('Load CSV data service ready ✅')

        self._load_pdf_data_srv = self._node.create_service(
            LoadPdfData, self._node.get_name()+'/load_pdf_data',
            self.load_pdf_data_callback)
        self._logger.info('Load PDF data service ready ✅')

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
        self._node.destroy_service(self._load_pdf_data_srv)
        self._node.destroy_service(self._save_index_srv)
        self._node.destroy_service(self._query_srv)
        self._node.destroy_service(self._rag_query_srv)

        self._logger.info('Load data service shutdown ✔️')
        self._logger.info('Load PDF data service shutdown ✔️')
        self._logger.info('Save index service shutdown ✔️')
        self._logger.info('Query service shutdown ✔️')
        self._logger.info('RAG query service shutdown ✔️')

        return True

    def get_history_callback(self, request, response):
        self._logger.info('GET HISTORY request received...')

        # Get & fill summary
        summary = self._conversation_history_manager.get_summary()
        response.history_summary.user_goals = summary['user_goals']
        response.history_summary.constraints = summary['constraints']
        response.history_summary.context = summary['context']

        # Get recent interactions
        _, queries, completions = (
            self._conversation_history_manager.get_recent_interactions())
        # Fill & insert recent interaction msgs
        for query, completion in zip(queries, completions):
            interaction = Interaction()
            interaction.query = query
            interaction.completion = completion

            response.recent_interactions.append(interaction)

        # Get evicted interactions
        _, queries, completions = (
            self._conversation_history_manager.get_evicted_interactions())
        # Fill & insert evicted interaction msgs
        for query, completion in zip(queries, completions):
            interaction = Interaction()
            interaction.query = query
            interaction.completion = completion

            response.evicted_interactions.append(interaction)

        # Fill header
        response.success = True
        response.error_code = 0
        response.error_msg = ''

        return response

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

    def load_pdf_data_callback(self, request, response):
        self._logger.info('LOAD PDF DATA request received...')

        # Get initial index size
        _, initial_size = self._retriever.get_index_size()

        # Load CSV data
        self._logger.info(f"Loading '{request.folder_path}'...")

        if not self._retriever.add_pdfs_from_folder(
                request.folder_path,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap):
            response.success = False
            response.error_code = -1
            response.error_msg = 'Error loading PDFs from folder'

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

    def _clean_query(self, completion: str, query: str) -> str:
        # Check if completion includes query
        if (query in completion and
                completion.find(query) == 0):
            clean_completion = completion[len(query):]
            # Clean initial chartacters
            clean_completion = clean_completion.lstrip(' .\n\t')
            # Clean bullets abd numerical headers if required
            if self._format_remove_bullets:
                clean_completion = re.sub(
                    r'^\s*\d+\s*[.)-]\s*',
                    '',
                    clean_completion,
                    flags=re.MULTILINE
                )

        else:
            clean_completion = completion

        return clean_completion

    def _remove_incomplete_sentences(self, completion: str) -> str:
        # Remove until last dot
        clean_completion = re.sub(r"\.[^.]*$", ".", completion)
        # Remove possible "numerations" (e.g. '2.')
        clean_completion = re.sub(r'[\n\s]*\d+\.\s*$', '', clean_completion)

        return clean_completion

    def _update_summary(self):
        # Get evicted queries/completions
        res, queries, completions = (
            self._conversation_history_manager.get_evicted_interactions()
        )

        # Check if evicted interactions stored
        if not res:
            # Release lock
            self._summary_lock.release()
            return

        # Build summary query
        summary_query = 'Extract the user goals, constraints, and context of' \
            ' this conversation. Do not add formatting. Do not use numbering' \
            ' or bullets. Use short, atomic statements.\n\n'
        summary_query += 'Interactions:\n'

        # Insert queries & completions
        for query, completion in zip(queries, completions):
            summary_query += 'User: ' + query + '\n'
            summary_query += 'Assistant: ' + completion + '\n'

        summary_query += '\nProvide a list of user goals, a list of ' \
            'constraints set by the user, and a list of relevant context' \
            ' of the conversation'

        # Get completion
        completion = ''

        self._generator_lock.acquire()

        res, completion = self._generator.generate(
            summary_query,
            self._gen_max_new_tokens,
            self._gen_temperature,
            self._gen_top_p)

        self._generator_lock.release()

        # Clean completion
        completion = self._clean_query(completion, summary_query)

        # Store summary
        self._conversation_history_manager.store_summary(completion)

        # Release lock
        self._summary_lock.release()

    def query_callback(self, request, response):
        self._logger.info('QUERY request received...')

        # If history active - Insert history prompt
        if self._history_active:
            query = (
                self._conversation_history_manager.get_history_prompt() +
                request.query + '\nAssistant: ')
        else:
            query = request.query

        # Get completion
        completion = ''

        self._generator_lock.acquire()

        res, completion = self._generator.generate(
            query,
            self._gen_max_new_tokens,
            self._gen_temperature,
            self._gen_top_p)

        self._generator_lock.release()

        if res:
            # Remove query if required
            if request.return_answer_only is True:
                completion = self._clean_query(completion, query)
            # Remove incomplete sentences if required
            if request.remove_incomplete_sentences is True:
                completion = self._remove_incomplete_sentences(completion)

            # Manage history if required
            if self._history_active:
                # Store query/completion
                self._conversation_history_manager.store_new_interaction(
                    request.query, completion
                )

                # Check if ther RAG system is already summarizing interactions
                if self._summary_lock.acquire(blocking=False):
                    # Update summary (separate thread to speed up response)
                    thread = threading.Thread(target=self._update_summary)
                    thread.start()
                else:
                    self._logger.info('Skipping summarization')

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

    def _verify_query_template(self, query_template: str,
                               history_active: bool) -> Tuple[bool, str]:
        # Check context tag
        if query_template.count('%context%') != 1:
            error_msg = ("The query template must contain a '%context%' tag " +
                         "to insert the information from the knowledge base")

            return False, error_msg
        # Check query tag
        elif query_template.count('%query%') != 1:
            error_msg = ("The query template must contain a '%query%' tag to" +
                         " insert the provided query")

            return False, error_msg
        # Check conversation summary tag (if applicable)
        elif (history_active and
              query_template.count('%conversation_summary%') > 1):
            error_msg = ("The query template contains more than a " +
                         "'%conversation_summary%' tag to insert the " +
                         "conversation summary")

            return False, error_msg
        # Check last user queries tag (if applicable)
        elif (history_active and
              query_template.count('%last_user_queries%') > 1):
            error_msg = ("The query template must contains more than a " +
                         "'%last_user_queries%' tag to insert the " +
                         "last user queries")

            return False, error_msg

        return True, ''

    def rag_query_callback(self, request, response):
        self._logger.info('RAG QUERY request received...')

        # Verify template
        res, error_msg = self._verify_query_template(request.query_template,
                                                     self._history_active)

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
            [request.query], self._retriever_top_k, self._retriever_alpha)

        # Check if context
        if not res:
            response.completion = ''
            response.success = False
            response.error_code = -3
            response.error_msg = 'Error retrieving context'

            self._logger.error('❌ ' + response.error_msg)

            return response

        # Insert context
        augmented_prompt = request.query_template.replace('%context%',
                                                          "\n".join(
                                                              contexts[0]))
        # Insert query
        augmented_prompt = augmented_prompt.replace('%query%', request.query)

        # Insert conversation summary if history active and tag present
        if (self._history_active and
                augmented_prompt.count('%conversation_summary%') == 1):
            # Insert summary
            conversation_summary = (
                self._conversation_history_manager
                .get_conversation_summary_as_string())
            augmented_prompt = augmented_prompt.replace(
                '%conversation_summary%', conversation_summary)
        # Insert last user queries if history active and tag present
        if (self._history_active and
                augmented_prompt.count('%last_user_queries%') == 1):
            # Insert summary
            last_user_queries = (
                self._conversation_history_manager
                .get_last_user_queries_as_string())
            augmented_prompt = augmented_prompt.replace(
                '%last_user_queries%', last_user_queries)

        # Get completion
        completion = ''

        self._generator_lock.acquire()

        res, completion = self._generator.generate(
            augmented_prompt,
            self._gen_max_new_tokens,
            self._gen_temperature,
            self._gen_top_p)

        self._generator_lock.release()

        if res:
            # Remove query if required
            if request.return_answer_only is True:
                completion = self._clean_query(completion, augmented_prompt)

            # Remove incomplete sentences if required
            if request.remove_incomplete_sentences is True:
                completion = self._remove_incomplete_sentences(completion)

            # Verify claims/sentences if required
            if self._rag_verification_active:
                # Get verified completion & removed claims
                res, completion, response.removed_claims = (
                    self._rag_verification_manager.verify_RAG_completion(
                        completion,
                        contexts[0],
                        self._rag_verification_entailment_threshold
                    )
                )

            # Manage history if required
            if self._history_active:
                # Store query/completion
                self._conversation_history_manager.store_new_interaction(
                    request.query, completion
                )

                # Check if ther RAG system is already summarizing interactions
                if self._summary_lock.acquire(blocking=False):
                    # Update summary (separate thread to speed up response)
                    thread = threading.Thread(target=self._update_summary)
                    thread.start()
                else:
                    self._logger.info('Skipping summarization')

            response.completion = completion
            response.context_chunks = contexts[0]
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
