# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-distutils python3.10-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip and install basic tools
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Move to ubuntu user home
WORKDIR /home/ubuntu/

# Clone llm_wrappers
RUN git clone https://github.com/aitor-ibarguren/llm_wrappers.git

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install llm_wrappers
RUN cd llm_wrappers && pip install --no-cache-dir -e .