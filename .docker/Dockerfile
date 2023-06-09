FROM osrf/ros:humble-desktop-jammy

SHELL ["/bin/bash", "-c", "-o", "pipefail"]

RUN echo "deb [trusted=yes] https://raw.githubusercontent.com/moveit/moveit2_packages/jammy-humble/ ./" \
    | sudo tee /etc/apt/sources.list.d/moveit_moveit2_packages.list
RUN echo "yaml https://raw.githubusercontent.com/moveit/moveit2_packages/jammy-humble/local.yaml humble" \
    | sudo tee /etc/ros/rosdep/sources.list.d/1-moveit_moveit2_packages.list

# Bring the container up to date
RUN apt-get update && apt-get upgrade -y && rosdep update

# Install packages required to run the demo
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-moveit \
    wget \
    python3-pip

# TODO: Make this more clean.
RUN apt-get install -y --no-install-recommends \
    ros-humble-moveit-resources-panda-moveit-config \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge \
    ros-humble-controller-manager

WORKDIR /app

COPY grab_anything ./grab_anything
COPY README.md requirements.txt setup.py ./grab_anything/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install setuptools wheel

RUN cd grab_anything && pip install .
RUN pip install git+https://github.com/huggingface/transformers

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Set up the entrypoint for both container start and interactive terminals.
COPY ./.docker/ros_entrypoint.sh /opt/.ros/
RUN echo "source /opt/.ros/ros_entrypoint.sh" >> ~/.bashrc
ENTRYPOINT [ "/opt/.ros/ros_entrypoint.sh" ]
