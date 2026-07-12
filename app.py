#!/usr/bin/env python3
"""
Voice Metadata Builder - web UI

A small Flask app that wraps the transcription/dedupe/split/training
pipeline in a browser UI, meant to run as a container with your
CharacterExtractor output folder mounted in.

Environment variables (set these in the TrueNAS app config):
    DATA_ROOT             Path inside the container to the "output" folder
                           (default: /data)
    WHISPER_URL            Base URL of the speaches server
                           (default: http://192.168.1.2:8014)
    WHISPER_MODEL           Model name speaches has loaded
                           (default: Systran/faster-whisper-large-v3)
    TRAINING_OUTPUT_ROOT   Where trained checkpoints and cached base XTTS
                           files are written (default: /training_output)
"""

import csv
import difflib
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))
WHISPER_URL = os.environ.get("WHISPER_URL", "http://192.168.1.2:8014")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "Systran/faster-whisper-large-v3")
TRAINING_OUTPUT_ROOT = Path(os.environ.get("TRAINING_OUTPUT_ROOT", "/training_output"))
TRAIN_TEMPLATE_PATH = Path(__file__).parent / "train_template.py"

# In-memory job tracking: job_id -> {status, log: [...], results: [...]}
JOBS = {}
JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Core logic (same behavior as the standalone script, refactored for reuse)
# ---------------------------------------------------------------------------

def get_duration(p: Path):
    try:
        import soundfile as sf
        info = sf.info(str(p))
        return info.frames / info.samplerate
    except Exception:
        return None


def transcribe_clip(wav_path: Path, timeout: int = 60):
    endpoint = f"{WHISPER_URL.rstrip('/')}/v1/audio/transcriptions"
    try:
        with open(wav_path, "rb") as f:
            files = {"file": (wav_path.name, f, "audio/wav")}
            data = {"model": WHISPER_MODEL}
            resp = requests.post(endpoint, files=files, data=data, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        text = payload.get("text", "").strip()
        return text if text else None
    except Exception as e:
        return None


def normalize_for_dedupe(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())


def is_near_duplicate(a: str, b: str, threshold: float = 0.92) -> bool:
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def read_metadata(csv_path: Path):
    """
    Reads the LJSpeech-style metadata.csv as actually written by
    export_xtts.py: three pipe-delimited fields per row --
        wav_id|transcript|normalized_transcript
    where wav_id has NO "wavs/" prefix and NO ".wav" extension (e.g.
    "Belldandy_00003", not "wavs/Belldandy_00003.wav"). The actual file
    lives at dataset_dir / "wavs" / f"{wav_id}.wav".

    We keep "normalized_transcript" in sync with "transcript" (writing
    the same value to both) since nothing downstream currently needs
    them to differ.
    """
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|")
        for line in reader:
            if not line or not line[0].strip():
                continue
            wav_id = line[0].strip()
            text = line[1].strip() if len(line) > 1 else ""
            rows.append({"wav_id": wav_id, "text": text})
    return rows


def wav_path_for(dataset_dir: Path, wav_id: str) -> Path:
    return dataset_dir / "wavs" / f"{wav_id}.wav"


def write_metadata(csv_path: Path, rows):
    """Writes back in the same 3-field LJSpeech format we read."""
    with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, delimiter="|", lineterminator="\n")
        for r in rows:
            writer.writerow([r["wav_id"], r["text"], r["text"]])


def log(job_id, message):
    with JOBS_LOCK:
        JOBS[job_id]["log"].append(message)


def run_transcription_job(job_id, dataset_dir: Path, force: bool, min_dur: float, max_dur: float):
    try:
        csv_path = dataset_dir / "metadata.csv"
        rows = read_metadata(csv_path)
        log(job_id, f"Loaded {len(rows)} rows from metadata.csv")

        # Early sanity check: if literally none of the referenced wav files
        # exist, something is wrong with paths/naming -- warn loudly instead
        # of silently grinding through 283 "missing" rows and wiping the file.
        if rows:
            sample_missing = sum(
                1 for r in rows[:20] if not wav_path_for(dataset_dir, r["wav_id"]).exists()
            )
            if sample_missing == min(20, len(rows)):
                log(job_id, f"[WARNING] None of the first {min(20, len(rows))} wav files were found on disk.")
                log(job_id, f"[WARNING] Expected path pattern: {wav_path_for(dataset_dir, rows[0]['wav_id'])}")
                log(job_id, "[WARNING] Check that the wavs/ folder and filenames match metadata.csv before continuing.")

        total = len(rows)
        done = 0
        for row in rows:
            wav_path = wav_path_for(dataset_dir, row["wav_id"])
            if not wav_path.exists():
                done += 1
                with JOBS_LOCK:
                    JOBS[job_id]["progress"] = done / total if total else 1
                continue
            if row["text"] and not force:
                done += 1
                with JOBS_LOCK:
                    JOBS[job_id]["progress"] = done / total if total else 1
                continue

            text = transcribe_clip(wav_path)
            if text is None:
                log(job_id, f"[FAIL] {row['wav_id']}")
            else:
                row["text"] = text
                log(job_id, f"[OK] {row['wav_id']} -> {text}")

            done += 1
            with JOBS_LOCK:
                JOBS[job_id]["progress"] = done / total if total else 1

        # Drop rows with missing wav or empty text, apply duration filter
        valid = []
        for r in rows:
            wp = wav_path_for(dataset_dir, r["wav_id"])
            if not wp.exists() or not r["text"]:
                continue
            dur = get_duration(wp)
            if dur is not None and (dur < min_dur or dur > max_dur):
                log(job_id, f"[DROP duration={dur:.2f}s] {r['wav_id']}")
                continue
            valid.append(r)

        if rows and not valid:
            log(job_id, "[ERROR] No valid rows survived processing -- refusing to overwrite metadata.csv "
                        "with an empty file. Fix the underlying issue (see warnings above) and try again.")
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
            return

        write_metadata(csv_path, valid)
        log(job_id, f"Wrote {len(valid)} valid rows back to metadata.csv")

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["results"] = valid
            JOBS[job_id]["progress"] = 1.0
    except Exception as e:
        log(job_id, f"[ERROR] Job failed: {e}")
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"


def run_split_job(job_id, dataset_dir: Path, dedupe_threshold: float, eval_fraction: float, seed: int):
    try:
        csv_path = dataset_dir / "metadata.csv"
        rows = read_metadata(csv_path)
        rows = [r for r in rows if r["text"] and wav_path_for(dataset_dir, r["wav_id"]).exists()]
        log(job_id, f"Starting from {len(rows)} valid rows")

        normalized = [normalize_for_dedupe(r["text"]) for r in rows]
        keep_mask = [True] * len(rows)
        for i in range(len(rows)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(rows)):
                if not keep_mask[j]:
                    continue
                if is_near_duplicate(normalized[i], normalized[j], dedupe_threshold):
                    keep_mask[j] = False
                    log(job_id, f"[DEDUPE] dropping {rows[j]['wav_id']} ~= {rows[i]['wav_id']}")

        deduped = [r for r, k in zip(rows, keep_mask) if k]
        log(job_id, f"After dedupe: {len(deduped)} rows")
        write_metadata(csv_path, deduped)

        random.seed(seed)
        shuffled = deduped[:]
        random.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * eval_fraction)) if shuffled else 0
        eval_rows = shuffled[:n_eval]
        train_rows = shuffled[n_eval:]

        write_metadata(dataset_dir / "metadata_train.csv", train_rows)
        write_metadata(dataset_dir / "metadata_eval.csv", eval_rows)
        log(job_id, f"Wrote metadata_train.csv ({len(train_rows)} rows) and metadata_eval.csv ({len(eval_rows)} rows)")

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["results"] = {"train": len(train_rows), "eval": len(eval_rows)}
            JOBS[job_id]["progress"] = 1.0
    except Exception as e:
        log(job_id, f"[ERROR] Job failed: {e}")
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"


def run_training_job(job_id, character_name: str, dataset_dir: Path,
                      num_epochs: int, batch_size: int, grad_accum_steps: int, language: str):
    """
    Builds a per-run copy of train_template.py with placeholders filled in,
    then runs it as a subprocess so we can stream stdout into the job log
    live (Coqui's trainer prints progress/loss per step to stdout).
    """
    try:
        train_csv = dataset_dir / "metadata_train.csv"
        eval_csv = dataset_dir / "metadata_eval.csv"
        if not train_csv.exists() or not eval_csv.exists():
            log(job_id, "[ERROR] metadata_train.csv / metadata_eval.csv not found. Run the split step first.")
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
            return

        # Reference clip for test-sentence generation during training.
        char_dir = dataset_dir.parent
        ref_dir = char_dir / "reference_clips"
        ref_wavs = sorted(ref_dir.glob("*.wav")) if ref_dir.exists() else []
        if not ref_wavs:
            log(job_id, "[ERROR] No reference_clips/*.wav found for this character.")
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
            return
        speaker_reference = str(ref_wavs[0])

        run_out_path = TRAINING_OUTPUT_ROOT / character_name / f"run_{int(time.time())}"
        run_out_path.mkdir(parents=True, exist_ok=True)

        template = TRAIN_TEMPLATE_PATH.read_text()
        filled = (
            template
            .replace("__CHARACTER_NAME__", character_name)
            .replace("__OUT_PATH__", str(run_out_path))
            .replace("__DATASET_PATH__", str(dataset_dir))
            .replace("__SPEAKER_REFERENCE__", speaker_reference)
            .replace("__NUM_EPOCHS__", str(num_epochs))
            .replace("__BATCH_SIZE__", str(batch_size))
            .replace("__GRAD_ACUMM_STEPS__", str(grad_accum_steps))
            .replace("__LANGUAGE__", language)
        )

        script_path = run_out_path / "train_run.py"
        script_path.write_text(filled)
        log(job_id, f"Wrote training script to {script_path}")
        log(job_id, f"Output dir: {run_out_path}")
        log(job_id, f"Reference clip: {speaker_reference}")
        log(job_id, "Starting training subprocess (this can take a long time)...")

        with JOBS_LOCK:
            JOBS[job_id]["run_out_path"] = str(run_out_path)

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with JOBS_LOCK:
            JOBS[job_id]["pid"] = proc.pid

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(job_id, line)
            # Best-effort progress parsing from Coqui's "EPOCH: n/N" style output.
            m = re.search(r"EPOCH:\s*(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
            if m:
                cur, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    with JOBS_LOCK:
                        JOBS[job_id]["progress"] = min(cur / total, 0.99)

        proc.wait()

        if proc.returncode != 0:
            log(job_id, f"[ERROR] Training process exited with code {proc.returncode}")
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
            return

        # Find the best/last checkpoint Coqui wrote and copy it to a stable,
        # predictable location for xtts-api-server to pick up.
        checkpoints = sorted(run_out_path.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
        config_files = sorted(run_out_path.rglob("config.json"), key=lambda p: p.stat().st_mtime)
        vocab_files = sorted(run_out_path.rglob("vocab.json"), key=lambda p: p.stat().st_mtime)

        if not checkpoints:
            log(job_id, "[ERROR] Training finished but no checkpoint (.pth) was found.")
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
            return

        final_dir = TRAINING_OUTPUT_ROOT / character_name / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoints[-1], final_dir / "model.pth")
        if config_files:
            shutil.copy2(config_files[-1], final_dir / "config.json")
        if vocab_files:
            shutil.copy2(vocab_files[-1], final_dir / "vocab.json")
        # Also drop the reference clip alongside it for convenience.
        shutil.copy2(speaker_reference, final_dir / "reference.wav")

        log(job_id, f"Copied final checkpoint to {final_dir}")
        log(job_id, "Point xtts-api-server's model path at this folder to use the fine-tuned voice.")

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["results"] = {"final_model_dir": str(final_dir)}
            JOBS[job_id]["progress"] = 1.0

    except Exception as e:
        log(job_id, f"[ERROR] Job failed: {e}")
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/characters")
def api_characters():
    """Scan DATA_ROOT for <Character>/finetune_dataset folders."""
    characters = []
    if not DATA_ROOT.exists():
        return jsonify({"error": f"DATA_ROOT {DATA_ROOT} does not exist", "characters": []})

    for child in sorted(DATA_ROOT.iterdir()):
        if not child.is_dir():
            continue
        ds = child / "finetune_dataset"
        csv_path = ds / "metadata.csv"
        wavs_dir = ds / "wavs"
        if not csv_path.exists():
            continue

        rows = read_metadata(csv_path)
        total = len(rows)
        transcribed = sum(1 for r in rows if r["text"])
        missing_wav = sum(1 for r in rows if not wav_path_for(ds, r["wav_id"]).exists())
        has_split = (ds / "metadata_train.csv").exists()
        final_model_dir = TRAINING_OUTPUT_ROOT / child.name / "final_model"
        has_trained_model = (final_model_dir / "model.pth").exists()

        characters.append({
            "name": child.name,
            "total_clips": total,
            "transcribed": transcribed,
            "missing_wav": missing_wav,
            "has_split": has_split,
            "has_trained_model": has_trained_model,
        })

    return jsonify({"characters": characters, "data_root": str(DATA_ROOT)})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    body = request.get_json()
    char_name = body["character"]
    force = bool(body.get("force", False))
    min_dur = float(body.get("min_duration", 1.0))
    max_dur = float(body.get("max_duration", 15.0))

    dataset_dir = DATA_ROOT / char_name / "finetune_dataset"
    if not dataset_dir.exists():
        return jsonify({"error": "character not found"}), 404

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "progress": 0.0, "results": None, "type": "transcribe"}

    thread = threading.Thread(
        target=run_transcription_job,
        args=(job_id, dataset_dir, force, min_dur, max_dur),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/split", methods=["POST"])
def api_split():
    body = request.get_json()
    char_name = body["character"]
    dedupe_threshold = float(body.get("dedupe_threshold", 0.92))
    eval_fraction = float(body.get("eval_fraction", 0.08))
    seed = int(body.get("seed", 42))

    dataset_dir = DATA_ROOT / char_name / "finetune_dataset"
    if not dataset_dir.exists():
        return jsonify({"error": "character not found"}), 404

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "progress": 0.0, "results": None, "type": "split"}

    thread = threading.Thread(
        target=run_split_job,
        args=(job_id, dataset_dir, dedupe_threshold, eval_fraction, seed),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/train", methods=["POST"])
def api_train():
    body = request.get_json()
    char_name = body["character"]
    num_epochs = int(body.get("num_epochs", 10))
    batch_size = int(body.get("batch_size", 4))
    grad_accum_steps = int(body.get("grad_accum_steps", 16))
    language = body.get("language", "en")

    dataset_dir = DATA_ROOT / char_name / "finetune_dataset"
    if not dataset_dir.exists():
        return jsonify({"error": "character not found"}), 404

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "progress": 0.0, "results": None, "type": "train"}

    thread = threading.Thread(
        target=run_training_job,
        args=(job_id, char_name, dataset_dir, num_epochs, batch_size, grad_accum_steps, language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def api_job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        # shallow copy so we don't hold the lock while serializing
        return jsonify({
            "status": job["status"],
            "log": job["log"],
            "progress": job["progress"],
            "results": job["results"],
            "type": job["type"],
        })


@app.route("/api/whisper-check")
def api_whisper_check():
    """Quick connectivity check the UI can call to show a green/red indicator."""
    try:
        resp = requests.get(f"{WHISPER_URL.rstrip('/')}/v1/models", timeout=5)
        resp.raise_for_status()
        return jsonify({"ok": True, "models": resp.json()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
#!/usr/bin/env python3
"""
Voice Metadata Builder - web UI

A small Flask app that wraps the transcription/dedupe/split/training
pipeline in a browser UI, meant to run as a container with your
CharacterExtractor output folder mounted in.

Environment variables (set these in the TrueNAS app config):
    DATA_ROOT             Path inside the container to the "output" folder
                           (default: /data)
    WHISPER_URL            Base URL of the speaches server
                           (default: http://192.168.1.2:8014)
    WHISPER_MODEL           Model name speaches has loaded
                           (default: Systran/faster-whisper-large-v3)
    TRAINING_OUTPUT_ROOT   Where trained checkpoints and cached base XTTS
                           files are written (default: /training_output)
"""

import csv
import difflib
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))
WHISPER_URL = os.environ.get("WHISPER_URL", "http://192.168.1.2:8014")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "Systran/faster-whisper-large-v3")
TRAINING_OUTPUT_ROOT = Path(os.environ.get("TRAINING_OUTPUT_ROOT", "/training_output"))
TRAIN_TEMPLATE_PATH = Path(__file__).parent / "train_template.py"

# In-memory job tracking: job_id -> {status, log: [...], results: [...]}
JOBS = {}
JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Core logic (same behavior as the standalone script, refactored for reuse)
# ---------------------------------------------------------------------------

def get_duration(p: Path):
    try:
        import soundfile as sf
        info = sf.info(str(p))
        return info.frames / info.samplerate
    except Exception:
        return None


def transcribe_clip(wav_path: Path, timeout: int = 60):
    endpoint = f"{WHISPER_URL.rstrip('/')}/v1/audio/transcriptions"
    try:
        with open(wav_path, "rb") as f:
            files = {"file": (wav_path.name, f, "audio/wav")}
            data = {"model": WHISPER_MODEL}
            resp = requests.post(endpoint, files=files, data=data, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        text = payload.get("text", "").strip()
        return text if text else None
    except Exception as e:
        return None


def normalize_for_dedupe(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())


def is_near_duplicate(a: str, b: str, threshold: float = 0.92) -> bool:
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def read_metadata(csv_path: Path):
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|")
        for line in reader:
            if not line:
                continue
            wav_rel = line[0].strip()
            text = line[1].strip() if len(line) > 1 else ""
            rows.append({"wav_rel": wav_rel, "text": text})
    return rows


def write_metadata(csv_path: Path, rows):
    with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, delimiter="|", lineterminator="\n")
        for r in rows:
            writer.writerow([r["wav_rel"], r["text"]])


def log(job_id, message):
    with JOBS_LOCK:
        JOBS[job_id]["log"].append(message)


def run_transcription_job(job_id, dataset_dir: Path, force: bool, min_dur: float, max_dur: float):
    try:
        csv_path = dataset_dir / "metadata.csv"
        rows = read_metadata(csv_path)
        log(job_id, f"Loaded {len(rows)} rows from metadata.csv")

        total = len(rows)
        done = 0
        for row in rows:
            wav_path = dataset_dir / row["wav_rel"]
            if not wav_path.exists():
                done += 1
                with JOBS_LOCK:
                    JOBS[job_id]["progress"] = done / total if total else 1
                continue
            if row["text"] and not force:
                done += 1
                with JOBS_LOCK:
                    JOBS[job_id]["progress"] = done / total if total else 1
                continue

            text = transcribe_clip(wav_path)
            if text is None:
                log(job_id, f"[FAIL] {row['wav_rel']}")
            else:
                row["text"] = text
                log(job_id, f"[OK] {row['wav_rel']} -> {text}")

            done += 1
            with JOBS_LOCK:
                JOBS[job_id]["progress"] = done / total if total else 1

        # Drop rows with missing wav or empty text, apply duration filter
        valid = []
        for r in rows:
            wp = dataset_dir / r["wav_rel"]
            if not wp.exists() or not r["text"]:
                continue
            dur = get_duration(wp)
            if dur is not None and (dur < min_dur or dur > max_dur):
                log(job_id, f"[DROP duration={dur:.2f}s] {r['wav_rel']}")
                continue
            valid.append(r)

        write_metadata(csv_path, valid)
        log(job_id, f"Wrote {len(valid)} valid rows back to metadata.csv")

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["results"] = valid
            JOBS[job_id]["progress"] = 1.0
    except Exception as e:
        log(job_id, f"[ERROR] Job failed: {e}")
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"


def run_split_job(job_id, dataset_dir: Path, dedupe_threshold: float, eval_fraction: float, seed: int):
    try:
        csv_path = dataset_dir / "metadata.csv"
        rows = read_metadata(csv_path)
        rows = [r for r in rows if r["text"] and (dataset_dir / r["wav_rel"]).exists()]
        log(job_id, f"Starting from {len(rows)} valid rows")

        normalized = [normalize_for_dedupe(r["text"]) for r in rows]
        keep_mask = [True] * len(rows)
        for i in range(len(rows)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(rows)):
                if not keep_mask[j]:
                    continue
                if is_near_duplicate(normalized[i], normalized[j], dedupe_threshold):
                    keep_mask[j] = False
                    log(job_id, f"[DEDUPE] dropping {rows[j]['wav_rel']} ~= {rows[i]['wav_rel']}")

        deduped = [r for r, k in zip(rows, keep_mask) if k]
        log(job_id, f"After dedupe: {len(deduped)} rows")
        write_metadata(csv_path, deduped)

        random.seed(seed)
        shuffled = deduped[:]
        random.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * eval_fraction)) if shuffled else 0
        eval_rows = shuffled[:n_eval]
        train_rows = shuffled[n_eval:]

        write_metadata(dataset_dir / "metadata_train.csv", train_rows)
        write_metadata(dataset_dir / "metadata_eval.csv", eval_rows)
        log(job_id, f"Wrote metadata_train.csv ({len(train_rows)} rows) and metadata_eval.csv ({len(eval_rows)} rows)")

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["results"] = {"train": len(train_rows), "eval": len(eval_rows)}
            JOBS[job_id]["progress"] = 1.0
    except Exception as e:
        log(job_id, f"[ERROR] Job failed: {e}")
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"


def run_training_job(job_id, character_name: str, dataset_dir: Path,
                      num_epochs: int, batch_size: int, grad_accum_steps: int, language: str):
    """
    Builds a per-run copy of train_template.py with placeholders filled in,
    then runs it as a subprocess so we can stream stdout into the job log
    live (Coqui's trainer prints progress/loss per step to stdout).
    """
    try:
        train_csv = dataset_dir / "metadata_train.csv"
        eval_csv = dataset_dir / "metadata_eval.csv"
        if not train_csv.exists() or not eval_csv.exists():
            log(job_id, "[ERROR] metadata_train.csv / metadata_eval.csv not found. Run the split step first.")
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
            return

        # Reference clip for test-sentence generation during training.
        char_dir = dataset_dir.parent
        ref_dir = char_dir / "reference_clips"
        ref_wavs = sorted(ref_dir.glob("*.wav")) if ref_dir.exists() else []
        if not ref_wavs:
            log(job_id, "[ERROR] No reference_clips/*.wav found for this character.")
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
            return
        speaker_reference = str(ref_wavs[0])

        run_out_path = TRAINING_OUTPUT_ROOT / character_name / f"run_{int(time.time())}"
        run_out_path.mkdir(parents=True, exist_ok=True)

        template = TRAIN_TEMPLATE_PATH.read_text()
        filled = (
            template
            .replace("__CHARACTER_NAME__", character_name)
            .replace("__OUT_PATH__", str(run_out_path))
            .replace("__DATASET_PATH__", str(dataset_dir))
            .replace("__SPEAKER_REFERENCE__", speaker_reference)
            .replace("__NUM_EPOCHS__", str(num_epochs))
            .replace("__BATCH_SIZE__", str(batch_size))
            .replace("__GRAD_ACUMM_STEPS__", str(grad_accum_steps))
            .replace("__LANGUAGE__", language)
        )

        script_path = run_out_path / "train_run.py"
        script_path.write_text(filled)
        log(job_id, f"Wrote training script to {script_path}")
        log(job_id, f"Output dir: {run_out_path}")
        log(job_id, f"Reference clip: {speaker_reference}")
        log(job_id, "Starting training subprocess (this can take a long time)...")

        with JOBS_LOCK:
            JOBS[job_id]["run_out_path"] = str(run_out_path)

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with JOBS_LOCK:
            JOBS[job_id]["pid"] = proc.pid

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(job_id, line)
            # Best-effort progress parsing from Coqui's "EPOCH: n/N" style output.
            m = re.search(r"EPOCH:\s*(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
            if m:
                cur, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    with JOBS_LOCK:
                        JOBS[job_id]["progress"] = min(cur / total, 0.99)

        proc.wait()

        if proc.returncode != 0:
            log(job_id, f"[ERROR] Training process exited with code {proc.returncode}")
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
            return

        # Find the best/last checkpoint Coqui wrote and copy it to a stable,
        # predictable location for xtts-api-server to pick up.
        checkpoints = sorted(run_out_path.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
        config_files = sorted(run_out_path.rglob("config.json"), key=lambda p: p.stat().st_mtime)
        vocab_files = sorted(run_out_path.rglob("vocab.json"), key=lambda p: p.stat().st_mtime)

        if not checkpoints:
            log(job_id, "[ERROR] Training finished but no checkpoint (.pth) was found.")
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
            return

        final_dir = TRAINING_OUTPUT_ROOT / character_name / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoints[-1], final_dir / "model.pth")
        if config_files:
            shutil.copy2(config_files[-1], final_dir / "config.json")
        if vocab_files:
            shutil.copy2(vocab_files[-1], final_dir / "vocab.json")
        # Also drop the reference clip alongside it for convenience.
        shutil.copy2(speaker_reference, final_dir / "reference.wav")

        log(job_id, f"Copied final checkpoint to {final_dir}")
        log(job_id, "Point xtts-api-server's model path at this folder to use the fine-tuned voice.")

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["results"] = {"final_model_dir": str(final_dir)}
            JOBS[job_id]["progress"] = 1.0

    except Exception as e:
        log(job_id, f"[ERROR] Job failed: {e}")
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/characters")
def api_characters():
    """Scan DATA_ROOT for <Character>/finetune_dataset folders."""
    characters = []
    if not DATA_ROOT.exists():
        return jsonify({"error": f"DATA_ROOT {DATA_ROOT} does not exist", "characters": []})

    for child in sorted(DATA_ROOT.iterdir()):
        if not child.is_dir():
            continue
        ds = child / "finetune_dataset"
        csv_path = ds / "metadata.csv"
        wavs_dir = ds / "wavs"
        if not csv_path.exists():
            continue

        rows = read_metadata(csv_path)
        total = len(rows)
        transcribed = sum(1 for r in rows if r["text"])
        missing_wav = sum(1 for r in rows if not (ds / r["wav_rel"]).exists())
        has_split = (ds / "metadata_train.csv").exists()
        final_model_dir = TRAINING_OUTPUT_ROOT / child.name / "final_model"
        has_trained_model = (final_model_dir / "model.pth").exists()

        characters.append({
            "name": child.name,
            "total_clips": total,
            "transcribed": transcribed,
            "missing_wav": missing_wav,
            "has_split": has_split,
            "has_trained_model": has_trained_model,
        })

    return jsonify({"characters": characters, "data_root": str(DATA_ROOT)})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    body = request.get_json()
    char_name = body["character"]
    force = bool(body.get("force", False))
    min_dur = float(body.get("min_duration", 1.0))
    max_dur = float(body.get("max_duration", 15.0))

    dataset_dir = DATA_ROOT / char_name / "finetune_dataset"
    if not dataset_dir.exists():
        return jsonify({"error": "character not found"}), 404

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "progress": 0.0, "results": None, "type": "transcribe"}

    thread = threading.Thread(
        target=run_transcription_job,
        args=(job_id, dataset_dir, force, min_dur, max_dur),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/split", methods=["POST"])
def api_split():
    body = request.get_json()
    char_name = body["character"]
    dedupe_threshold = float(body.get("dedupe_threshold", 0.92))
    eval_fraction = float(body.get("eval_fraction", 0.08))
    seed = int(body.get("seed", 42))

    dataset_dir = DATA_ROOT / char_name / "finetune_dataset"
    if not dataset_dir.exists():
        return jsonify({"error": "character not found"}), 404

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "progress": 0.0, "results": None, "type": "split"}

    thread = threading.Thread(
        target=run_split_job,
        args=(job_id, dataset_dir, dedupe_threshold, eval_fraction, seed),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/train", methods=["POST"])
def api_train():
    body = request.get_json()
    char_name = body["character"]
    num_epochs = int(body.get("num_epochs", 10))
    batch_size = int(body.get("batch_size", 4))
    grad_accum_steps = int(body.get("grad_accum_steps", 16))
    language = body.get("language", "en")

    dataset_dir = DATA_ROOT / char_name / "finetune_dataset"
    if not dataset_dir.exists():
        return jsonify({"error": "character not found"}), 404

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "progress": 0.0, "results": None, "type": "train"}

    thread = threading.Thread(
        target=run_training_job,
        args=(job_id, char_name, dataset_dir, num_epochs, batch_size, grad_accum_steps, language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def api_job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        # shallow copy so we don't hold the lock while serializing
        return jsonify({
            "status": job["status"],
            "log": job["log"],
            "progress": job["progress"],
            "results": job["results"],
            "type": job["type"],
        })


@app.route("/api/whisper-check")
def api_whisper_check():
    """Quick connectivity check the UI can call to show a green/red indicator."""
    try:
        resp = requests.get(f"{WHISPER_URL.rstrip('/')}/v1/models", timeout=5)
        resp.raise_for_status()
        return jsonify({"ok": True, "models": resp.json()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
