# Vivijure Wan 2.2 A14B character-LoRA TRAINING image (vivijure-wan-train satellite).
#
# Decoupled from vivijure-backend (Conrad ruling 2026-07-23). Trains dual-expert Wan character
# LoRAs via ai-toolkit; uploads to R2/MinIO at loras/<project>/<slot>/wan_{high,low}_noise.safetensors.
#
# Two isolated conda envs (cf#29 D1): "vivijure" (job handler + R2) and "aitoolkit" (ai-toolkit run.py).

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS deps

ARG AITOOLKIT_REF=6e158dd1f1552b73b7aca6d7ddaa46a783538052
ARG VJ_PYTHON_VERSION=3.11
ARG VJ_TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG VJ_TORCH_SPEC="torch==2.9.0 torchvision torchaudio"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/opt/wan-train/src \
    HF_HOME=/opt/models/hf-cache \
    VJ_MODELS_ROOT=/opt/models \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates unzip ffmpeg libgl1 libglib2.0-0 build-essential openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda \
    && rm /tmp/mc.sh \
    && conda create -y -n vivijure -c conda-forge --override-channels "python=${VJ_PYTHON_VERSION}" pip \
    && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "vivijure", "/bin/bash", "-c"]
RUN python -m pip install --upgrade pip wheel setuptools \
    && python -m pip install --index-url "${VJ_TORCH_INDEX}" ${VJ_TORCH_SPEC}
COPY deploy/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

SHELL ["/bin/bash", "-c"]
RUN conda create -y -n aitoolkit -c conda-forge --override-channels "python=${VJ_PYTHON_VERSION}" pip && conda clean -afy
RUN conda run --no-capture-output -n aitoolkit python -m pip install --upgrade pip wheel setuptools \
    && conda run --no-capture-output -n aitoolkit python -m pip install --index-url "${VJ_TORCH_INDEX}" torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
RUN git clone https://github.com/ostris/ai-toolkit /opt/ai-toolkit \
    && git -C /opt/ai-toolkit checkout ${AITOOLKIT_REF}
COPY deploy/aitoolkit-constraints.txt /tmp/aitoolkit-constraints.txt
COPY deploy/aitoolkit-overrides.txt /tmp/aitoolkit-overrides.txt
RUN conda run --no-capture-output -n aitoolkit python -m pip install \
        -c /tmp/aitoolkit-constraints.txt -r /opt/ai-toolkit/requirements.txt \
    && conda run --no-capture-output -n aitoolkit python -m pip install \
        -c /tmp/aitoolkit-constraints.txt -r /tmp/aitoolkit-overrides.txt

ENV VIVIJURE_AITOOLKIT_DIR=/opt/ai-toolkit \
    VIVIJURE_AITOOLKIT_PYTHON=/opt/conda/envs/aitoolkit/bin/python

# --- STAGE SPLIT (#29) ---------------------------------------------------------
# Everything ABOVE is the dependency surface: apt, both conda envs, the cu128 torch trio,
# deploy/requirements.txt, the ai-toolkit checkout and its constraints/overrides. Everything
# BELOW is the ~55GB Wan weight bake plus the worker code.
#
# The dispatch-gated GPU compat smoke (.github/workflows/compat-smoke.yml) builds this
# stage by name (docker build --target deps), so it exercises THESE layers, the real
# production ones, without staging 55GB of weights it does not need to answer a
# dependency question. A production build names no target and therefore still builds
# the full stage, unchanged.
#
# Keep every dependency install ABOVE this line; tests/test_compat_smoke_contract.py fails CI
# if a pip/conda/apt install drifts below it (a smoke that misses a dep layer is a smoke
# that lies).
FROM deps AS full

COPY deploy/train-bins/bin-00/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-01/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-02/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-03/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-04/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-05/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-06/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-07/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-08/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-09/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-10/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-11/ /opt/models/hf-cache/
COPY deploy/train-bins/weights-manifest.sha256 /opt/models/hf-cache/weights-manifest.sha256
RUN cd /opt/models/hf-cache \
    && sha256sum -c weights-manifest.sha256 --quiet \
    && echo "wan base bake: sha256 verified ($(wc -l < weights-manifest.sha256) files)." \
    && rm -f weights-manifest.sha256

RUN conda run --no-capture-output -n aitoolkit python -c "from huggingface_hub import snapshot_download; [print('offline-resolve OK:', r, snapshot_download(r, local_files_only=True)) for r in ('ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16','ai-toolkit/umt5_xxl_encoder','ai-toolkit/wan2.1-vae')]"

COPY deploy/bake_aitoolkit_offline_paths.py /tmp/bake_aitoolkit_offline_paths.py
RUN conda run --no-capture-output -n aitoolkit python /tmp/bake_aitoolkit_offline_paths.py \
    && rm -f /tmp/bake_aitoolkit_offline_paths.py
ENV VIVIJURE_AITOOLKIT_WEIGHTS=/opt/models/aitoolkit \
    VIVIJURE_WAN_BASE_PATH=/opt/models/aitoolkit/wan-base

COPY deploy/smoke_train_offline.py /tmp/smoke_train_offline.py
RUN conda run --no-capture-output -n aitoolkit python /tmp/smoke_train_offline.py \
    && rm -f /tmp/smoke_train_offline.py

WORKDIR /opt/wan-train
COPY src/wan_train /opt/wan-train/src/wan_train
COPY handler.py /opt/wan-train/handler.py

RUN conda run --no-capture-output -n vivijure python -c "from wan_train import wan_lora_train as W; assert W.aitoolkit_python().endswith('/aitoolkit/bin/python'), W.aitoolkit_python(); assert W.default_wan_base_path().startswith('/opt/models/aitoolkit/'), W.default_wan_base_path(); print('wan_train import OK')"
RUN conda run --no-capture-output -n aitoolkit python -c "import torch, torchaudio, diffusers, transformers, safetensors; print('aitoolkit import smoke OK:', torch.__version__)"

CMD ["conda", "run", "--no-capture-output", "-n", "vivijure", "python", "-u", "/opt/wan-train/handler.py"]
