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
    # ros-humble-moveit \
    wget \
    python3-pip


# TODO: Remove as we moved them to grab_anything.
# RUN git clone -b panda_gz --depth 1 --single-branch https://github.com/henningkayser/moveit_resources.git
# RUN git clone -b panda_gz --depth 1 --single-branch https://github.com/henningkayser/moveit2_tutorials.git
#TODO(henningkayser) remove when #2222 and #2223 are merged
# RUN git clone -b panda_gz --depth 1 --single-branch https://github.com/henningkayser/moveit2.git
# COPY ./ moveit2_tutorials

WORKDIR /app

COPY README.md requirements.txt ./
COPY grab_anything ./grab_anything

# Install dependencies
RUN pip install --upgrade pip
RUN pip install setuptools wheel

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/huggingface/transformers

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create Colcon workspace and clone the needed source code to run the demo
RUN mkdir -p /moveit_ws/src
WORKDIR /app/moveit_ws/src
RUN cd /moveit_ws/src && \
    git clone -b panda_gz --depth 1 --single-branch https://github.com/henningkayser/moveit_resources.git && \
    git clone -b panda_gz --depth 1 --single-branch https://github.com/henningkayser/moveit2_tutorials.git && \
    git clone -b panda_gz --depth 1 --single-branch https://github.com/henningkayser/moveit2.git && \
    cd .. && \
    source /opt/ros/humble/setup.bash && \
    apt-get update -y && \
    rosdep install --from-paths src --ignore-src --rosdistro "$ROS_DISTRO" -y && \
    rm -rf /var/lib/apt/lists/* && \
    source /opt/ros/humble/setup.bash && colcon build --symlink-install --parallel-workers 1

WORKDIR /app

# Clone repo, initialize and update submodules, and download pretrained weights
RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git && \
    cd Grounded-Segment-Anything && \
    git submodule init && \
    git submodule update && \
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && \
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && \
    cd Tag2Text && \
    wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth && \
    cd ../segment_anything && pip install . && \
    cd ../GroundingDINO && pip install . && \
    cd ../Tag2Text && git submodule update --init --recursive && pip install -r requirements.txt

# Set up the entrypoint for both container start and interactive terminals.
COPY ./.docker/ros_entrypoint.sh /opt/.ros/
RUN echo "source /opt/.ros/ros_entrypoint.sh" >> ~/.bashrc
ENTRYPOINT [ "/opt/.ros/ros_entrypoint.sh" ]
