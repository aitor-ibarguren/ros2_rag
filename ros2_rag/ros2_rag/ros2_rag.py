import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from ros2_rag._ros2_rag_class import ROS2RAGClass


class ROS2RAGNode(LifecycleNode):

    def __init__(self, node_name='ros2_rag', ns='', options=None):
        super().__init__(node_name, namespace=ns)

        # Declare parameters
        self.declare_parameter('auto_activate', True)

        self.get_logger().info("Class ROS2RAGNode initialized")

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(
            f"Node '{self.get_name()}' is in state '{state.label}'. "
            "Transitioning to 'inactive'"
        )

        # Declare ROS2 RAG class
        self._ros2_rag = ROS2RAGClass(self)

        # Define LLM-RAG params
        params = {}
        params['generator_type'] = 'flan_t5'
        params['generator_loading'] = 'pretrained'

        # Configure
        try:
            if self._ros2_rag.configure(params):
                # Successful configuration
                self.get_logger().info('Successful configuration 🛠️')
                return TransitionCallbackReturn.SUCCESS
            else:
                # Error in configuration
                self.get_logger().error('❌ Failure in configuration')
                return TransitionCallbackReturn.FAILURE 
        except Exception as e:
            # Error in configuration
            self.get_logger().error(f"❌ Failure in configuration: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(
            f"Node '{self.get_name()}' is in state '{state.label}'. "
            "Transitioning to 'unconfigured'"
        )

        # Destroy ROS2 RAG class
        del self._ros2_rag

        # Finish transition
        self.get_logger().info("Successful cleanup")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(
            f"Node '{self.get_name()}' is in state '{state.label}'. "
            "Transitioning to 'active'"
        )

        # Activate
        try:
            if self._ros2_rag.activate():
                # Successful activation
                self.get_logger().info('Successful activation 🚀')
                return TransitionCallbackReturn.SUCCESS
            else:
                # Error in activation
                self.get_logger().error('❌ Failure in activation')
                return TransitionCallbackReturn.FAILURE
        except Exception as e:
            # Error in activation
            self.get_logger().error(f"❌ Failure in activation: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(
            f"Node '{self.get_name()}' is in state '{state.label}'. "
            "Transitioning to 'configured'"
        )

        # Deactivate
        try:
            if self._ros2_rag.deactivate():
                # Successful deactivation
                self.get_logger().info('Successful deactivation')
                return TransitionCallbackReturn.SUCCESS
            else:
                # Error in deactivation
                self.get_logger().error('❌ Failure in deactivation')
                return TransitionCallbackReturn.FAILURE
        except Exception as e:
            # Error in deactivation
            self.get_logger().error(f"❌ Failure in deactivation: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(
            f"Node '{self.get_name()}' is in state '{state.label}'. "
            "Transitioning to 'shutdown'"
        )

        # Finish transition
        self.get_logger().info("Transition finished!")
        return TransitionCallbackReturn.SUCCESS


def main():
    rclpy.init()

    # Create execute
    executor = rclpy.executors.SingleThreadedExecutor()

    # Create node and add to executor
    ros2_rag_node = ROS2RAGNode('ros2_rag')
    executor.add_node(ros2_rag_node)

    # Get the parameter value
    auto_activate = ros2_rag_node.get_parameter(
        'auto_activate').get_parameter_value().bool_value

    # Configure and activate node if required
    if auto_activate:
        ros2_rag_node.trigger_configure()
        ros2_rag_node.trigger_activate()

    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        ros2_rag_node.destroy_node()


if __name__ == '__main__':
    main()
