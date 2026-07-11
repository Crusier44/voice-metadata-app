FROM ghcr.io/idiap/coqui-tts:latest

# The base image ships Coqui TTS + PyTorch + CUDA already wired up for GPU use.
# We add our Flask control-panel app on top of it, and swap the entrypoint
# from the base image's `tts` CLI to our web app.

WORKDIR /app

# Skip the interactive CPML license prompt (XTTS v2 model license) on first load.
ENV COQUI_TOS_AGREED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY train_template.py .
COPY templates/ templates/

EXPOSE 5000

ENTRYPOINT []
CMD ["python3", "app.py"]
