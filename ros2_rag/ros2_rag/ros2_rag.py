import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from ros2_rag._ros2_rag_class import ROS2RAGClass
from ros2_rag_msgs.srv import Query


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

        # Declare PSD class
        self._ros2_rag = ROS2RAGClass(self.get_logger())

        # Finish transition
        self.get_logger().info("Transition finished!")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(
            f"Node '{self.get_name()}' is in state '{state.label}'. "
            "Transitioning to 'unconfigured'"
        )

        # Destroy PSD class
        del self._ros2_rag

        # Finish transition
        self.get_logger().info("Transition finished!")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(
            f"Node '{self.get_name()}' is in state '{state.label}'. "
            "Transitioning to 'active'"
        )

        # Initialize services
        self._load_data_srv = self.create_service(
            Query, self.get_name()+'/query', self._ros2_rag.query)

        # Finish transition
        self.get_logger().info("Transition finished!")
        return TransitionCallbackReturn.SUCCESS

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
