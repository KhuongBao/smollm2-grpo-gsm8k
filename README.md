# GRPO Fine-Tuning: SmolLM2-360M on GSM8K

Compares SFT against four GRPO reward variants (R1–R4) on grade-school math reasoning using [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) and the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset. Training runs on a single A100 via Google Colab with LoRA (rank 16) and vLLM acceleration.

**Full write-up:** See [report.pdf](report.pdf) for methodology, results, and analysis.

## Usage

```bash
python main.py
```

**Or step through interactively in Colab using `GRPO.ipynb`.**

## Requirements

```
trl[vllm]
transformers
datasets
peft
torch
regex
numpy
pandas
matplotlib
seaborn
```

```bash
pip install -r requirements.txt
```

## Structure

```
├── config.py          # All paths and hyperparameters
├── dataset.py         # GSM8K loading and preprocessing
├── model.py           # Model/tokenizer loading and chat inference
├── rewards.py         # R1–R4 reward functions
├── train_sft.py       # SFT training loop
├── train_grpo.py      # GRPO training loop 
├── evaluate.py        # Evaluation metrics and result plotting
├── main.py            # End-to-end pipeline script
└── GRPO.ipynb         # Original interactive Colab notebook
```

## Results

| Model   | Accuracy | Format Correct | Avg Total Len | Avg Think Len |
|---------|----------|----------------|---------------|---------------|
| SFT     | 4.17%    | 41.50%         | 241.8         | 81.7          |
| GRPO-R1 | 3.83%    | 39.67%         | 243.5         | 76.3          |
| GRPO-R2 | 4.17%    | 38.67%         | 246.0         | 79.1          |
| GRPO-R3 | 3.00%    | 40.33%         | 243.9         | 78.4          |
| GRPO-R4 | 2.50%    | 38.17%         | 245.4         | 66.9          |

SFT was the strongest baseline. GRPO variants did not improve accuracy at this scale, with R4's length penalty visibly shortening `<think>` output without a correctness benefit.

## Reward Functions

| Variant | Name | Description |
|---------|------|-------------|
| R1 | Baseline | +1 if predicted answer matches gold, else 0 |
| R2 | Format + Outcome | +0.2 for `<think>` tags, +0.2 for `<answer>` tags, +0.6 for correct answer |
| R3 | Reasoning Aware | Up to +0.3 for arithmetic steps inside `<think>` (0.05/step), +0.7 for correct answer |
| R4 | Efficiency Regularized | +0.8 for correct answer minus a length penalty for completions over 150 tokens |


