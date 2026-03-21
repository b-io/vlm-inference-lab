# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python package
RUN pip3 install --no-cache-dir -e .

# Set default command
CMD ["bash"]
