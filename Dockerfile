# RunPod PyTorch base image with CUDA 12.4 + Python 3.11.
# NOTE: do not switch to a Blackwell (sm_120) GPU — this cu12.4 toolchain
# tops out at sm_90. Endpoint must keep "PRO 6000 MIG 24GB" UNCHECKED and
# use L4 (Ada, sm_89). See README "Deploy" for the exact console setting.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies (VoxCPM needs ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (pinned — see requirements.txt)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Worker code
COPY handler.py /app/handler.py

# Caches land on the network volume with the model; offline mode is set in
# handler.py before any HF import. Keep HF_HOME off container disk.
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV TORCH_HOME=/runpod-volume/huggingface-cache/torch
ENV TOKENIZERS_PARALLELISM=false

CMD ["python3", "-u", "/app/handler.py"]
