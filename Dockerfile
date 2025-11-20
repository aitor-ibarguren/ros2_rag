FROM ros:jazzy

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install git and colcon build tools
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    && apt-get clean

# Create ROS2 workspace
ENV ROS2_WS=/home/ubuntu/ros2_ws
RUN mkdir -p $ROS2_WS/src

# Set working directory
WORKDIR $ROS2_WS/src

# Clone repo & submodules
RUN git clone --recursive https://github.com/aitor-ibarguren/ros2_rag.git
# Update submodulke
RUN cd ros2_rag && git submodule update --remote --recursive

# Install requirements of llm_wrappers
RUN cd ./ros2_rag/ros2_rag/llm_wrappers && pip install -r requirements.txt --break-system-packages

# Go back to workspace folder
WORKDIR $ROS2_WS

# Install ROS2 dependencies
RUN rosdep update && rosdep install --from-paths src -y --ignore-src

# Source ROS2 & build workspace
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build"