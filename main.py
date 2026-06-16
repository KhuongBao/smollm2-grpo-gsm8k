"""
Pipeline:
    Load + preprocess GSM8K
    Train SFT baseline
    Train GRPO R1-R4
    Evaluate all models
    Plot results
"""


from config import SAVE_PATHS
from dataset import load_gsm8k, get_sft_datasets, get_grpo_dataset
from model import load_base_model, load_model_from_path, save_model
from train_sft import train_sft
from train_grpo import train_all_grpo_variants
from evaluate import evaluate, print_results, plot_results


def main():
    dataset = load_gsm8k()
    train_dataset, test_dataset = get_sft_datasets(dataset)
    grpo_dataset = get_grpo_dataset(dataset)

    model, tokenizer = load_base_model()
    save_model(model, tokenizer, SAVE_PATHS["base"])

    sft_model = train_sft(model, tokenizer, train_dataset)

    sft_model, tokenizer = load_model_from_path(SAVE_PATHS["sft"])
    train_all_grpo_variants(sft_model, tokenizer, grpo_dataset)

    base_model, _ = load_model_from_path(SAVE_PATHS["base"])
    sft_model, tokenizer = load_model_from_path(SAVE_PATHS["sft"])
    grpo_r1, _ = load_model_from_path(SAVE_PATHS["grpo_r1"])
    grpo_r2, _ = load_model_from_path(SAVE_PATHS["grpo_r2"])
    grpo_r3, _ = load_model_from_path(SAVE_PATHS["grpo_r3"])
    grpo_r4, _ = load_model_from_path(SAVE_PATHS["grpo_r4"])

    all_results = {
        "Base":    evaluate(base_model,  tokenizer, test_dataset, "Base Model"),
        "SFT":     evaluate(sft_model,   tokenizer, test_dataset, "SFT Model"),
        "GRPO-R1": evaluate(grpo_r1,     tokenizer, test_dataset, "GRPO Model R1"),
        "GRPO-R2": evaluate(grpo_r2,     tokenizer, test_dataset, "GRPO Model R2"),
        "GRPO-R3": evaluate(grpo_r3,     tokenizer, test_dataset, "GRPO Model R3"),
        "GRPO-R4": evaluate(grpo_r4,     tokenizer, test_dataset, "GRPO Model R4"),
    }

    print_results(all_results)
    plot_results(all_results)


if __name__ == "__main__":
    main()
