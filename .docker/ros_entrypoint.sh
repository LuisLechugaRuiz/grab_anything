#!/bin/bash

# Source ROS 2 Humble
source /opt/ros/humble/setup.bash
echo "Sourced ROS 2 Humble"

# Source the base workspace, if built
if [ -f /app/grab_anything_ws/install/setup.bash ]
then
  source /app/grab_anything_ws/install/setup.bash
  echo "Sourced grab anything workspace"
else
  echo "Please build the grab_anything_ws with:"
  echo "colcon build --symlink-install"
fi

# Execute the command passed into this entrypoint
exec "$@"
