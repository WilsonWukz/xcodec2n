# Dockerfile - for CUDA 12.4.1 + PyTorch 2.5.1
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=Etc/UTC

# 1) 安装系统依赖
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

# 2) 设置 Python 3.10 为默认 python，并升级 pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# 3) 安装 PyTorch 2.5.1+cu124 和相关库
RUN python -m pip install --no-cache-dir \
    "torch==2.5.1+cu124" \
    "torchvision==0.20.1+cu124" \
    "torchaudio==2.5.1+cu124" \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 4) 复制仓库文件
COPY . /workspace

# 5) 安装 requirements.txt 中的依赖
RUN if [ -f requirements.txt ]; then \
      python -m pip install --no-cache-dir -r requirements.txt ; \
    fi

# 6) 下载预训练模型（根据你的 README 调整）
RUN mkdir -p ckpt && \
    wget -O ckpt/epoch=4-step=1400000.ckpt https://huggingface.co/HKUST-Audio/xcodec2/resolve/main/ckpt/epoch%3D4-step%3D1400000.ckpt || true

# 7) 设置环境变量和默认命令
ENV PYTHONUNBUFFERED=1
CMD ["bash"]