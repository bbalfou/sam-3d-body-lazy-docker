# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    HYDRA_FULL_ERROR=1

ARG USERNAME=dev
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG MAX_JOBS=8

# Blackwell / sm_120
ARG CUDA_ARCH_LIST="12.0+PTX"

# Optional extras from the upstream install guide
ARG INSTALL_MOGE=false
ARG INSTALL_SAM3=false

ENV MAX_JOBS=${MAX_JOBS}
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}
ENV FORCE_CUDA=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    git \
    ffmpeg \
    build-essential \
    ninja-build \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    libosmesa6-dev freeglut3-dev libegl1-mesa-dev

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python

# Create non-root user matching host UID/GID to avoid bind-mount permission issues
RUN set -eux; \
    if ! getent group "${GROUP_ID}" >/dev/null; then \
        groupadd --gid "${GROUP_ID}" "${USERNAME}"; \
    fi; \
    if ! id -u "${USER_ID}" >/dev/null 2>&1; then \
        useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --shell /bin/bash "${USERNAME}"; \
    fi; \
    mkdir -p /workspace; \
    chown -R "${USER_ID}:${GROUP_ID}" /workspace /home/"${USERNAME}"

USER ${USERNAME}
ENV HOME=/home/${USERNAME}
WORKDIR /workspace

# Virtualenv
RUN python3.11 -m venv ${HOME}/.venv
ENV VIRTUAL_ENV=${HOME}/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

# Base Python tooling
RUN pip install --upgrade pip wheel "setuptools<81" ninja psutil \
 && pip install "numpy<2.0.0"

# PyTorch: official CUDA 12.8 wheels
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Python dependencies from the SAM 3D Body guide
RUN pip install \
    pytorch-lightning \
    pyrender \
    opencv-python \
    yacs \
    scikit-image \
    einops \
    timm \
    dill \
    pandas \
    rich \
    hydra-core \
    hydra-submitit-launcher \
    hydra-colorlog \
    pyrootutils \
    webdataset \
    chump \
    networkx==3.2.1 \
    roma \
    joblib \
    seaborn \
    wandb \
    appdirs \
    appnope \
    ffmpeg \
    cython \
    jsonlines \
    pytest \
    xtcocotools \
    loguru \
    optree \
    fvcore \
    black \
    pycocotools \
    tensorboard \
    huggingface_hub

# Detectron2 from source, as requested by upstream guide
RUN pip install \
    'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation \
    --no-deps

# Optional: MoGe
RUN if [ "${INSTALL_MOGE}" = "true" ]; then \
      pip install git+https://github.com/microsoft/MoGe.git; \
    fi

# Optional: minimal SAM3 install for inference support
RUN if [ "${INSTALL_SAM3}" = "true" ]; then \
      git clone https://github.com/facebookresearch/sam3.git ${HOME}/src/sam3 && \
      pip install -e ${HOME}/src/sam3 && \
      pip install decord psutil; \
    fi

CMD ["bash"]