import sys
import os

from trl import GRPOConfig, GRPOTrainer

from config import SAVE_PATHS, GRPO_CONFIG
from rewards import REWARD_FUNCTIONS


def _patch_stdout_fileno():
    """Patch sys.stdout.fileno to avoid UnsupportedOperation in Colab."""
    sys.stdout.fileno = lambda: os.open(os.devnull, os.O_WRONLY)


def train_grpo_variant(model, tokenizer, grpo_dataset, reward_key):
    """
    Train a single GRPO variant.

    Args:
        model:        The SFT-initialised model.
        tokenizer:    Matching tokenizer.
        grpo_dataset: Dataset formatted for GRPO (prompt + answer columns).
        reward_key:   One of "r1", "r2", "r3", "r4".
    """
    save_path = SAVE_PATHS[f"grpo_{reward_key}"]
    reward_func = REWARD_FUNCTIONS[reward_key]

    grpo_config = GRPOConfig(
        output_dir=save_path,
        **GRPO_CONFIG,
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=grpo_dataset,
    )

    grpo_trainer.train()

    grpo_trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    return grpo_trainer


def train_all_grpo_variants(model, tokenizer, grpo_dataset):
    """Train R1–R4 sequentially, returning a dict of trainers."""
    _patch_stdout_fileno()
    trainers = {}
    for key in REWARD_FUNCTIONS:
        print(f"\n{'='*50}\nTraining GRPO {key.upper()}\n{'='*50}")
        trainers[key] = train_grpo_variant(model, tokenizer, grpo_dataset, key)
    return trainers
