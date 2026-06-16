import torch

# ── Paths ────────────────────────────────────────────────────────────────────
DRIVE_PATH = "/content/drive/My Drive/Colab Notebooks/Deep RL/"

SAVE_PATHS = {
    "base":    DRIVE_PATH + "smollm2-base",
    "sft":     DRIVE_PATH + "smollm2-sft-gsm8k",
    "grpo_r1": DRIVE_PATH + "smollm2-grpo-r1",
    "grpo_r2": DRIVE_PATH + "smollm2-grpo-r2",
    "grpo_r3": DRIVE_PATH + "smollm2-grpo-r3",
    "grpo_r4": DRIVE_PATH + "smollm2-grpo-r4",
}

# ── Model ─────────────────────────────────────────────────────────────────────
CHECKPOINT = "HuggingFaceTB/SmolLM2-360M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── SFT ───────────────────────────────────────────────────────────────────────
SFT_CONFIG = dict(
    max_length=512,
    dataset_text_field="text",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    bf16=True,
)

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# ── GRPO ──────────────────────────────────────────────────────────────────────
GRPO_CONFIG = dict(
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    num_generations=4,
    max_completion_length=512,
    bf16=True,
    logging_steps=10,
    gradient_checkpointing=True,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.3,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_N = 600

CHAT_CONFIG = dict(
    max_new_tokens=100,
    do_sample=True,
    top_p=0.9,
    temperature=0.2,
)
