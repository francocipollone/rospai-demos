# Docker Development Guide

This guide explains how to use Docker containers for developing and running the ROS Physical AI Demos.

## Prerequisites

### Install Docker
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to the docker group (logout/login required)
sudo usermod -aG docker $USER

# Install Docker Compose (if not included)
sudo apt install docker-compose-plugin
```

### For NVIDIA GPU Support (Optional but Recommended)
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```


## Quick Start

There are two possible approaches for development:

1. **Using VS Code Dev Containers (Recommended for full IDE integration)**
   - Open this folder in VS Code and install the recommended Dev Container extension.
   - Reopen in container when prompted, or use the "Dev Containers: Reopen in Container" command.
   - The environment will be set up automatically, including user permissions and middleware configuration.

2. **Running the Container Manually via Scripts**
   - Use the provided build and run scripts as described below.
   - This approach is suitable for working outside of VS Code or for custom workflows.

### 1. Clone and Import Repositories (On Host)

Clone repositories on the host to keep source code accessible for version control and IDE integration:

```bash
mkdir ~/ws_pai/src -p && cd ~/ws_pai/src
git clone https://github.com/ros-physical-ai/demos
vcs import . < demos/pai.repos --recursive
```

### 2. Option 1: Using VS Code Dev Containers (Recommended)

Open this folder (ros-physical-ai/demos) in VS Code and select "Reopen in Container" when prompted. The development environment will be set up automatically, and you can start working inside the containerized environment immediately.


> **Note:** You are ready to go!


### 2. Option 2: Running the Container Manually via Scripts

#### 1. Build the docker image.

If you are using the manual scripts approach (Option 2), build the Docker image as follows. This ensures your host user's UID/GID are passed to the container for proper file permissions:

```bash
cd ~/ws_pai/src/demos
./docker/build.sh
```

Or manually set the environment variables:
```bash
cd ~/ws_pai/src/demos/docker
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build
```

> **Why is this important?** Files created inside the container need to be accessible by your host user and vice versa. By matching the container user's UID/GID to your host user, you avoid permission issues with mounted volumes.

#### 2. Allow X11 Connections (for GUI applications) (Option 2: Manual Container Usage Only)
```bash
# Run this on the host before starting the container
xhost +local:docker
```

> **Note:** For security, you can restrict this after your session:
> ```bash
> xhost -local:docker
> ```

#### 3. Start the Development Container (Option 2: Manual Container Usage Only)

Use the provided run script:
```bash
./docker/run.sh --rm pai-dev
```

Or manually:
```bash
cd ~/ws_pai/src/demos/docker
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d pai-dev
docker compose exec pai-dev bash
```

Or start interactively:
```bash
cd ~/ws_pai/src/demos/docker
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm pai-dev
```

## Development Workflow

### Initial Setup (Inside Container)

1. **Install dependencies:**
   ```bash
   cd /ros_ws
   rosdep install --from-paths src --ignore-src --rosdistro kilted -yir
   ```

2. **Build the workspace:**
   ```bash
   colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
   source install/setup.bash
   ```

### Running the Demo

```bash
# Inside the container
source /ros_ws/install/setup.bash
ros2 launch pai_bringup so_arm_gz_bringup.launch.py
```

## Using rmw_zenoh (Recommended)

For better ROS 2 communication performance, use the Zenoh middleware:

> **Note:** The development container is pre-configured to use `rmw_zenoh_cpp` as the default ROS 2 middleware implementation. You do not need to set `RMW_IMPLEMENTATION` manually unless you want to override this behavior.

### Option 1: Start Zenoh Router in Separate Container
```bash
# From the docker directory
cd ~/ws_pai/src/demos/docker
docker compose --profile zenoh up -d zenoh-router

# In your development container, set the middleware
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
```

### Option 2: Run Zenoh Router on another terminal
```bash
# Open another terminal in the container
cd ~/ws_pai/src/demos/docker
docker compose exec pai-dev bash

# Inside the container
ros2 run rmw_zenoh_cpp rmw_zenohd
```

## Common Tasks

### Opening Multiple Terminals
```bash
# In new host terminal windows
cd ~/ws_pai/src/demos/docker
docker compose exec pai-dev bash
```

### Rebuilding After Code Changes
```bash
# Inside container
cd /ros_ws
colcon build --packages-select <package_name>
source install/setup.bash
```

### Accessing Real Hardware

To connect to real hardware (e.g., SO-ARM100), uncomment the device mappings in `docker/docker-compose.yml`:

```yaml
devices:
  - /dev/dri:/dev/dri
  - /dev/ttyUSB0:/dev/ttyUSB0  # Uncomment this
  - /dev/ttyACM0:/dev/ttyACM0  # Uncomment this if needed
```

Then restart the container:
```bash
cd ~/ws_pai/src/demos/docker
docker compose down
docker compose up -d pai-dev
```

### Persistent Build Artifacts

Build artifacts are stored in bind-mounted directories to speed up subsequent builds:
- `.pai-build/` - Build directory
- `.pai-install/` - Install directory  
- `.pai-log/` - Log directory

To clean and start fresh:
```bash
cd ~/ws_pai/src/demos/docker
docker compose down
docker compose up -d pai-dev
```
