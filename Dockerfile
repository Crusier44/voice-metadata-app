FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV COQUI_TOS_AGREED=1

WORKDIR /app

# Ubuntu 22.04 ships Python 3.10 by default -- use that rather than
# installing 3.11 separately, since python3-pip only pairs correctly with
# whichever python3 is already the system default. Mixing versions here is
# what caused "pip: command not found" (exit 127) previously.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Confirm pip is actually on PATH before we rely on it below -- fails the
# build immediately with a clear message instead of a mysterious exit 127
# later if something about the base image ever changes.
RUN python3 -m pip --version

# PyTorch with CUDA 12.4 wheels, matching the base image's CUDA version.
RUN python3 -m pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Coqui TTS (community-maintained fork, PyPI package name is coqui-tts)
# plus its trainer package needed for the fine-tuning recipe.
RUN python3 -m pip install --no-cache-dir coqui-tts coqui-tts-trainer

# coqui-tts is not yet compatible with transformers 5.x -- it imports
# isin_mps_friendly from transformers.pytorch_utils, which was removed in
# that release (confirmed: https://github.com/idiap/coqui-ai-TTS/issues/558).
# coqui-tts's own dependency resolution pulls in a 5.x version regardless,
# so we force it back down afterward.
RUN python3 -m pip install --no-cache-dir "transformers>=4.57,<5"

# Our app's own lightweight dependencies (Flask, requests). soundfile is
# already pulled in transitively by coqui-tts, but listing it is harmless.
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY train_template.py .
COPY templates/ templates/

EXPOSE 5000

CMD ["python3", "app.py"]
