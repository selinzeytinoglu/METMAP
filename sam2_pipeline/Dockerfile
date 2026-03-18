# Base image with PyTorch and CUDA pre-installed.
# Requires host NVIDIA driver >= 550 (CUDA 12.4 compatible).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# System dependencies needed by OpenCV and SAM2's CUDA extension compiler
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    matplotlib \
    pillow \
    tqdm \
    hydra-core \
    iopath \
    pyyaml

# Copy SAM2 source and install as a package so Hydra can find configs
COPY sam2/ ./sam2/
COPY setup.py ./
RUN pip install --no-cache-dir -e .

# Download the SAM2 large checkpoint
RUN mkdir -p sam2/checkpoints && \
    wget -q -O sam2/checkpoints/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Copy scripts
COPY scripts/ ./scripts/

# /data is the mount point for the user's data (frames, prompts, output)
VOLUME ["/data"]

CMD ["python", "scripts/segment_video.py"]
