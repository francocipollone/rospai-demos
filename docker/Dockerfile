# ROS Physical AI Demos - Development Container
# Based on ROS 2 Kilted with Gazebo Ionic

ARG ROS_DISTRO=kilted

FROM osrf/ros:${ROS_DISTRO}-desktop-full

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up locale
RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install development tools and dependencies
RUN apt-get update && apt-get install -y \
    # Development tools
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    # GUI and visualization
    libgl1-mesa-dri \
    mesa-utils \
    x11-apps \
    # Gazebo Ionic
    ros-${ROS_DISTRO}-ros-gz \
    ros-${ROS_DISTRO}-gz-ros2-control \
    # ROS 2 Control
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-controller-manager \
    # Additional ROS packages
    ros-${ROS_DISTRO}-ament-cmake-vendor-package \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-rmw-zenoh-cpp \
    ros-${ROS_DISTRO}-rqt-joint-trajectory-controller \
    ros-${ROS_DISTRO}-nav2-common \
    ros-${ROS_DISTRO}-moveit-ros-move-group \
    ros-${ROS_DISTRO}-moveit-kinematics \
    ros-${ROS_DISTRO}-moveit-planners \
    ros-${ROS_DISTRO}-moveit-simple-controller-manager \
    ros-${ROS_DISTRO}-moveit-ros-control-interface \
    ros-${ROS_DISTRO}-moveit-ros-visualization \
    ros-${ROS_DISTRO}-moveit-ros-warehouse \
    ros-${ROS_DISTRO}-moveit-setup-assistant \
    # Other dependencies
    librange-v3-dev \
    libglfw3-dev \
    wayland-protocols \
    # Serial communication (for real hardware)
    libserial-dev \
    # Utilities
    vim \
    nano \
    wget \
    curl \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
RUN mkdir -p /ros_ws/src

WORKDIR /ros_ws

# Set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create a non-root user for development
# Dynamically handle UID/GID to match the host user
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install sudo first
RUN apt-get update && apt-get install -y sudo \
    && rm -rf /var/lib/apt/lists/*

# Handle user creation/modification based on UID/GID
# If the ubuntu user exists with different UID/GID, modify it
# Otherwise create the user
RUN if id -u $USERNAME > /dev/null 2>&1; then \
        # User exists, modify UID/GID if different \
        if [ "$(id -u $USERNAME)" != "$USER_UID" ]; then \
            usermod -u $USER_UID $USERNAME; \
        fi && \
        if [ "$(id -g $USERNAME)" != "$USER_GID" ]; then \
            groupmod -g $USER_GID $USERNAME; \
        fi; \
    else \
        # Create group if it doesn't exist \
        groupadd -o --gid $USER_GID $USERNAME && \
        # Create user \
        useradd -o --uid $USER_UID --gid $USER_GID -s /bin/bash -m $USERNAME; \
    fi \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Give the user ownership of the workspace
RUN chown -R $USERNAME:$USERNAME /ros_ws

# Add user to dialout group for serial port access
RUN usermod -aG dialout $USERNAME

# Set environment variables
ENV SHELL=/bin/bash

# Switch to non-root user
USER $USERNAME

# Update rosdep database
RUN rosdep update

# Add ROS setup to bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc \
    && echo "if [ -f /ros_ws/install/setup.bash ]; then source /ros_ws/install/setup.bash; fi" >> ~/.bashrc \
    && echo "# Colcon argcomplete" >> ~/.bashrc \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc \
    && echo "# Optional: Use rmw_zenoh as middleware" >> ~/.bashrc \
    && echo "# export RMW_IMPLEMENTATION=rmw_zenoh_cpp" >> ~/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
