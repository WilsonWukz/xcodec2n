# Dockerfile - for CUDA 12.4 + PyTorch 2.5.1
FROM nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=Etc/UTC

# 1) system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    wget \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# 2) Install CUDA-specific PyTorch wheels (cu124).
#    Use the official PyTorch cu124 index to ensure matching builds.
#    We pin torch==2.5.1+cu124 and torchvision==0.20.1+cu124 (2.5.1 <-> 0.20.1 pairing).
#    If you need a different torch version, change both torch & torchvision accordingly.
RUN python -m pip install --no-cache-dir \
    "torch==2.5.1+cu124" \
    "torchvision==0.20.1+cu124" \
    "torchaudio==2.5.1+cu124" \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 3) repository / project files
COPY . /workspace

# 4) Install python dependencies defined by your repo (if you have requirements)
#    Make sure requirements.txt doesn't pin conflicting torch/torchvision/torchaudio versions.
RUN if [ -f requirements.txt ]; then \
      python -m pip install --no-cache-dir -r requirements.txt ; \
    fi

# 5) optional: expose ports, set entrypoint, etc.
ENV PYTHONUNBUFFERED=1
CMD ["bash"]
