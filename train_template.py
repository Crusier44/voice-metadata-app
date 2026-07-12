"""
Template for XTTS v2 fine-tuning, adapted from the official Coqui recipe:
TTS/recipes/ljspeech/xtts_v2/train_gpt_xtts.py

This file is NOT run directly. The app copies it per training job and
substitutes the __PLACEHOLDER__ values before executing it, so each
character's training run gets its own config, dataset paths, and output dir
without editing this file by hand.

Placeholders substituted by run_training_job() in app.py:
    __CHARACTER_NAME__     e.g. "Belldandy"
    __OUT_PATH__            base output dir for this training run
    __DATASET_PATH__        the finetune_dataset dir (contains wavs/, metadata_train.csv, metadata_eval.csv)
    __SPEAKER_REFERENCE__   path to one reference wav (from reference_clips/) for test-sentence generation during training
    __NUM_EPOCHS__          int
    __BATCH_SIZE__          int
    __GRAD_ACUMM_STEPS__    int
    __LANGUAGE__            e.g. "en"
"""

import gc
import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
)
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

CHARACTER_NAME = "__CHARACTER_NAME__"
OUT_PATH = "__OUT_PATH__"
DATASET_PATH = "__DATASET_PATH__"
SPEAKER_REFERENCE = ["__SPEAKER_REFERENCE__"]
NUM_EPOCHS = __NUM_EPOCHS__
BATCH_SIZE = __BATCH_SIZE__
GRAD_ACUMM_STEPS = __GRAD_ACUMM_STEPS__
LANGUAGE = "__LANGUAGE__"

os.makedirs(OUT_PATH, exist_ok=True)

# Dataset in LJSpeech format: metadata_train.csv / metadata_eval.csv already
# written by the app's dedupe+split step, wav paths relative to DATASET_PATH.
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name=CHARACTER_NAME,
    path=DATASET_PATH,
    meta_file_train="metadata_train.csv",
    meta_file_val="metadata_eval.csv",
    language=LANGUAGE,
)
DATASETS_CONFIG_LIST = [config_dataset]

# Base XTTS v2 checkpoint files -- downloaded once, cached under OUT_PATH,
# reused across all future training runs so subsequent characters don't
# re-download several GB each time.
CHECKPOINTS_OUT_PATH = os.path.join(os.path.dirname(OUT_PATH.rstrip("/")), "XTTS_v2.0_original_model_files")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

if not all(os.path.isfile(p) for p in [DVAE_CHECKPOINT, MEL_NORM_FILE]):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files(
        [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

if not all(os.path.isfile(p) for p in [TOKENIZER_FILE, XTTS_CHECKPOINT]):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )


def main():
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,   # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,           # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=f"XTTS_finetune_{CHARACTER_NAME}",
        project_name="XTTS_finetune",
        run_description=f"XTTS v2 GPT fine-tuning for {CHARACTER_NAME}",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=4,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        eval_split_max_size=256,
        print_step=25,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        # test_sentences intentionally omitted: generating full audio each
        # epoch (via CPU/GPU inference) is a plausible source of the
        # per-epoch RAM growth observed on this box. Re-enable once training
        # is confirmed stable if you want live audio samples during training.
        epochs=NUM_EPOCHS,
    )

    model = GPTTrainer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.01,  # metadata_eval.csv already IS the eval split; keep this small
    )

    def free_memory_on_epoch_end(trainer_instance):
        """
        Forces Python garbage collection and clears the CUDA allocator
        cache after every epoch. Observed behavior on this box was a
        stair-step RAM increase roughly aligned with epoch/eval boundaries,
        eventually triggering an OOM kill -- this callback is a direct
        countermeasure for that pattern.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(" > [memory cleanup] gc.collect() + torch.cuda.empty_cache() ran after epoch end", flush=True)

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        callbacks={"on_epoch_end": free_memory_on_epoch_end},
    )
    trainer.fit()


if __name__ == "__main__":
    main()"""
Template for XTTS v2 fine-tuning, adapted from the official Coqui recipe:
TTS/recipes/ljspeech/xtts_v2/train_gpt_xtts.py

This file is NOT run directly. The app copies it per training job and
substitutes the __PLACEHOLDER__ values before executing it, so each
character's training run gets its own config, dataset paths, and output dir
without editing this file by hand.

Placeholders substituted by run_training_job() in app.py:
    __CHARACTER_NAME__     e.g. "Belldandy"
    __OUT_PATH__            base output dir for this training run
    __DATASET_PATH__        the finetune_dataset dir (contains wavs/, metadata_train.csv, metadata_eval.csv)
    __SPEAKER_REFERENCE__   path to one reference wav (from reference_clips/) for test-sentence generation during training
    __NUM_EPOCHS__          int
    __BATCH_SIZE__          int
    __GRAD_ACUMM_STEPS__    int
    __LANGUAGE__            e.g. "en"
"""

import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
)
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

CHARACTER_NAME = "__CHARACTER_NAME__"
OUT_PATH = "__OUT_PATH__"
DATASET_PATH = "__DATASET_PATH__"
SPEAKER_REFERENCE = ["__SPEAKER_REFERENCE__"]
NUM_EPOCHS = __NUM_EPOCHS__
BATCH_SIZE = __BATCH_SIZE__
GRAD_ACUMM_STEPS = __GRAD_ACUMM_STEPS__
LANGUAGE = "__LANGUAGE__"

os.makedirs(OUT_PATH, exist_ok=True)

# Dataset in LJSpeech format: metadata_train.csv / metadata_eval.csv already
# written by the app's dedupe+split step, wav paths relative to DATASET_PATH.
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name=CHARACTER_NAME,
    path=DATASET_PATH,
    meta_file_train="metadata_train.csv",
    meta_file_val="metadata_eval.csv",
    language=LANGUAGE,
)
DATASETS_CONFIG_LIST = [config_dataset]

# Base XTTS v2 checkpoint files -- downloaded once, cached under OUT_PATH,
# reused across all future training runs so subsequent characters don't
# re-download several GB each time.
CHECKPOINTS_OUT_PATH = os.path.join(os.path.dirname(OUT_PATH.rstrip("/")), "XTTS_v2.0_original_model_files")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

if not all(os.path.isfile(p) for p in [DVAE_CHECKPOINT, MEL_NORM_FILE]):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files(
        [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

if not all(os.path.isfile(p) for p in [TOKENIZER_FILE, XTTS_CHECKPOINT]):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )


def main():
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,   # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,           # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=f"XTTS_finetune_{CHARACTER_NAME}",
        project_name="XTTS_finetune",
        run_description=f"XTTS v2 GPT fine-tuning for {CHARACTER_NAME}",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=4,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        eval_split_max_size=256,
        print_step=25,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": f"Hello, this is a test of the {CHARACTER_NAME} voice model.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
        epochs=NUM_EPOCHS,
    )

    model = GPTTrainer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.01,  # metadata_eval.csv already IS the eval split; keep this small
    )

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
