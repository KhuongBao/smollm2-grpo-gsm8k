import regex as re
from datasets import load_dataset


def load_gsm8k():
    return load_dataset("openai/gsm8k", "main")


def preprocess_gsm8k(example):
    parts = example["answer"].split("####")
    thought_content = parts[0].strip()
    final_answer = parts[1].strip()

    thought_content = re.sub(r"<<.*?>>", "", thought_content)

    formatted_text = (
        f"Question: {example['question']}\n"
        f"<think>\n{thought_content}\n</think>\n"
        f"<answer>{final_answer}</answer>"
    )

    return {"text": formatted_text}


def format_for_grpo(example):
    parts = example["answer"].split("####")
    return {
        "prompt": f"Question: {example['question']}\n<think>\n",
        "answer": parts[1].strip(),
    }


def get_sft_datasets(dataset):
    train_dataset = dataset["train"].map(preprocess_gsm8k)
    test_dataset = dataset["test"].map(preprocess_gsm8k)
    return train_dataset, test_dataset


def get_grpo_dataset(dataset):
    return dataset["train"].map(
        format_for_grpo,
        remove_columns=dataset["train"].column_names,
    )
