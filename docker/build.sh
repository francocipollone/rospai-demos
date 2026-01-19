#!/bin/bash
# Build script that passes host user UID/GID to docker-compose
# This ensures file permissions work correctly between host and container

set -e

# Change to the directory containing this script (docker/)
cd "$(dirname "$0")"

export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

echo "Building with HOST_UID=$HOST_UID and HOST_GID=$HOST_GID"

docker compose build "$@"
