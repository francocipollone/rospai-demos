#!/bin/bash
# Run script that ensures correct UID/GID environment variables are set
# This is needed if the image was built with the correct UID/GID

set -e

# Change to the directory containing this script (docker/)
cd "$(dirname "$0")"

export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# Allow X11 forwarding (needed for GUI apps like Gazebo, RViz)
xhost +local:docker 2>/dev/null || true

docker compose run "$@"
