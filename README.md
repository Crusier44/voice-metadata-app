# Voice Metadata Builder

A small web UI wrapping the XTTS fine-tuning data pipeline: transcribe
character voice clips, dedupe/split into train/eval sets, and kick off
XTTS v2 fine-tuning -- all from a browser instead of a terminal.

## Pipeline stages (one page, one character at a time)

1. **Transcribe** -- fills in `metadata.csv` for a character by sending
   each wav clip to your speaches (Whisper) server.
2. **Dedupe & split** -- drops near-duplicate lines, writes
   `metadata_train.csv` / `metadata_eval.csv`.
3. **Train** -- runs Coqui's XTTS v2 GPT fine-tuning recipe as a background
   job, streams progress live, and copies the resulting checkpoint to
   `TRAINING_OUTPUT_ROOT/<character>/final_model/`.

## Expected input folder structure

```
output/
  Belldandy/
    reference_clips/       (a handful of clean reference wavs)
    finetune_dataset/
      metadata.csv          ("wavs/belldandy_0001.wav|")
      wavs/
        belldandy_0001.wav
        ...
```

## Deploying

See `truenas-compose.yaml` for the TrueNAS custom app config. In short:

1. Push this repo to GitHub.
2. GitHub Actions (`.github/workflows/build-push.yml`) builds the image and
   pushes it to `ghcr.io/<you>/<repo>:latest` automatically on every push
   to `main`.
3. In TrueNAS, install a custom app using `truenas-compose.yaml` as a guide
   (or paste it directly if your TrueNAS version supports YAML import),
   filling in your actual GitHub username/repo name and confirming the bind
   mount paths match your pool layout.
4. Open `http://<ryozanpaku-ip>:5010` (or whatever host port you chose).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DATA_ROOT` | `/data` | Mounted CharacterExtractor output folder |
| `WHISPER_URL` | `http://192.168.1.2:8014` | speaches server base URL |
| `WHISPER_MODEL` | `Systran/faster-whisper-large-v3` | Model name speaches has loaded |
| `TRAINING_OUTPUT_ROOT` | `/training_output` | Where checkpoints + cached base XTTS files land |

## Using the trained model

After training finishes, `TRAINING_OUTPUT_ROOT/<character>/final_model/`
contains `model.pth`, `config.json`, `vocab.json`, and a copy of the
reference clip used. Point your `xtts-api-server` deployment's model path
at this folder to serve the fine-tuned voice instead of the base model.
