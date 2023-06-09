FROM osrf/ros:humble-desktop-jammy

SHELL ["/bin/bash", "-c", "-o", "pipefail"]

ARG CUDA_MAJOR_VERSION=11
ARG CUDA_MINOR_VERSION=7

RUN echo "deb [trusted=yes] https://raw.githubusercontent.com/moveit/moveit2_packages/jammy-humble/ ./" \
    | sudo tee /etc/apt/sources.list.d/moveit_moveit2_packages.list
RUN echo "yaml https://raw.githubusercontent.com/moveit/moveit2_packages/jammy-humble/local.yaml humble" \
    | sudo tee /etc/ros/rosdep/sources.list.d/1-moveit_moveit2_packages.list

# Bring the container up to date
RUN apt-get update && apt-get upgrade -y && rosdep update

# Install packages required to run the demo
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-moveit \
    ros-humble-moveit-resources \
    wget \
    python3-pip

# TODO: Make this more clean.
RUN apt-get install -y --no-install-recommends \
    ros-humble-moveit-resources-panda-moveit-config \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge \
    ros-humble-controller-manager

# Install CUDA
ENV DEBIAN_FRONTEND noninteractive
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-cudart-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}  cuda-compat-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} cuda-toolkit-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

WORKDIR /app

COPY grab_anything ./grab_anything
COPY grab_anything_ws ./grab_anything_ws

# Install dependencies
RUN pip install --upgrade pip
RUN pip install setuptools wheel

RUN cd grab_anything && pip install .
RUN pip install git+https://github.com/huggingface/transformers

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app/grab_anything_ws/src
RUN git clone -b panda_gz --depth 1 --single-branch https://github.com/henningkayser/moveit2_tutorials.git

WORKDIR /app/grab_anything_ws/
RUN source /opt/ros/humble/setup.bash \
    && apt-get update -y \
    && rosdep install --from-paths src --ignore-src --rosdistro "$ROS_DISTRO" -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set up the entrypoint for both container start and interactive terminals.
COPY ./.docker/ros_entrypoint.sh /opt/.ros/
RUN echo "source /opt/.ros/ros_entrypoint.sh" >> ~/.bashrc
ENTRYPOINT [ "/opt/.ros/ros_entrypoint.sh" ]
