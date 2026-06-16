import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import EVAL_N
from model import chat


def evaluate(model, tokenizer, dataset, model_name, n=EVAL_N):
    correct = 0
    format_correct = 0
    lengths = []
    think_lengths = []

    print(f"\n--- Qualitative Sample for {model_name} ---")
    for i, example in enumerate(tqdm(dataset.select(range(n)), desc=f"Evaluating {model_name}")):
        response = chat(model, tokenizer, example["question"])
        gold = example["answer"].split("####")[1].strip()

        # Accuracy
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        predicted = match.group(1).strip() if match else ""
        if predicted == gold:
            correct += 1

        # Formatting heuristics
        has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
        has_answer = bool(match)
        if has_think and has_answer:
            format_correct += 1

        # Length / Efficiency
        lengths.append(len(response))
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        think_lengths.append(len(think_match.group(1)) if think_match else 0)

        # Qualitative sample
        if i == 0:
            print(f"Q: {example['question']}")
            print(f"Gold: {gold}")
            print(f"Model Output:\n{response}\n")

    return {
        "accuracy": correct / n,
        "format_correct": format_correct / n,
        "avg_total_length": np.mean(lengths),
        "avg_think_length": np.mean(think_lengths),
    }


def plot_results(all_results):
    df_results = pd.DataFrame.from_dict(all_results, orient="index")
    df_results.index.name = "Model"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Comparison of Model Performance Metrics (Log Scale)", fontsize=16)

    def format_log_axis(ax, title, ylabel, data_col, palette, format_str):
        sns.barplot(
            x=df_results.index,
            y=data_col,
            data=df_results,
            ax=ax,
            palette=palette,
            hue=df_results.index,
            legend=False,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.set_yscale("log")

        for p in ax.patches:
            val = p.get_height()
            if val > 0:
                ax.annotate(
                    f"{val:{format_str}}",
                    (p.get_x() + p.get_width() / 2.0, val),
                    ha="center",
                    va="bottom",
                    xytext=(0, 0),
                    textcoords="offset points",
                )

    format_log_axis(axes[0, 0], "Accuracy", "Accuracy (Log)", "accuracy", "viridis", ".4f")
    format_log_axis(axes[0, 1], "Format Correctness", "Format Correctness (Log)", "format_correct", "plasma", ".4f")
    format_log_axis(axes[1, 0], "Avg Total Length", "Length (Log)", "avg_total_length", "magma", ".1f")
    format_log_axis(axes[1, 1], "Avg Think Length", "Length (Log)", "avg_think_length", "cividis", ".1f")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def print_results(all_results):
    for name, res in all_results.items():
        print(
            f"{name}: acc={res['accuracy']:.4f} | format={res['format_correct']:.4f} | "
            f"avg_total_len={res['avg_total_length']:.1f} | avg_think_len={res['avg_think_length']:.1f}"
        )
