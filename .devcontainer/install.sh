#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

# sudo apt update
# sudo apt install -y libopenblas-dev libyaml-dev ffmpeg wget ca-certificates

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update
# sudo apt-get -y install cuda

# Install poetry
pipx install poetry --pip-args '--no-cache-dir --force-reinstall'

# Install project dependencies
poetry install
