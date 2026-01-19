#!/bin/bash
set -e

# Source ROS 2 setup
source /opt/ros/${ROS_DISTRO}/setup.bash

# Source workspace setup if available
if [ -f /ros_ws/install/setup.bash ]; then
    source /ros_ws/install/setup.bash
fi

# Execute the command passed to the container
exec "$@"
