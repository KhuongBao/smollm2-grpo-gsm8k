from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from config import SAVE_PATHS, SFT_CONFIG, LORA_CONFIG
from model import save_model


def train_sft(model, tokenizer, train_dataset):
    peft_config = LoraConfig(**LORA_CONFIG)

    sft_config = SFTConfig(
        output_dir=SAVE_PATHS["sft"],
        **SFT_CONFIG,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=sft_config,
    )

    trainer.train()

    merged_model = trainer.model.merge_and_unload()
    save_model(merged_model, tokenizer, SAVE_PATHS["sft"])

    return merged_model
