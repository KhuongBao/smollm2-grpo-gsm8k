"""
GRPO reward functions:

R1: Baseline
    +1 if predicted answer matches gold answer, else 0

R2: Format + Outcome
    +0.2 for <think> tags, +0.2 for <answer> tags, +0.6 for correct answer

R3: Reasoning Aware
    Up to +0.3 for arithmetic steps inside <think> (0.05 per step, capped),
    +0.7 for correct answer

R4: Efficiency Regularized
    +0.8 for correct answer minus a length penalty for completions > 150 tokens
"""

import regex as re


def reward_r1(completions, **kwargs):
    answers = kwargs.get("answer", [])
    rewards = []
    for completion, answer in zip(completions, answers):
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        predicted = match.group(1).strip() if match else ""
        rewards.append(1.0 if predicted == answer else 0.0)
    return rewards


def reward_r2(completions, **kwargs):
    answers = kwargs.get("answer", [])
    rewards = []
    for completion, answer in zip(completions, answers):
        score = 0.0
        if re.search(r"<think>.*?</think>", completion, re.DOTALL):
            score += 0.2
        if re.search(r"<answer>.*?</answer>", completion, re.DOTALL):
            score += 0.2
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match and match.group(1).strip() == answer.strip():
            score += 0.6
        rewards.append(score)
    return rewards


def reward_r3(completions, **kwargs):
    answers = kwargs["answer"]
    rewards = []
    for completion, gold in zip(completions, answers):
        score = 0.0
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if think_match:
            thought = think_match.group(1)
            num_steps = len(re.findall(r"\d+\s*[\+\-\*\/]\s*\d+", thought))
            score += min(0.3, num_steps * 0.05)
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match and match.group(1).strip() == gold.strip():
            score += 0.7
        rewards.append(score)
    return rewards


def reward_r4(completions, **kwargs):
    answers = kwargs["answer"]
    rewards = []
    for completion, gold in zip(completions, answers):
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        correct = match and match.group(1).strip() == gold.strip()
        length_penalty = max(0, (len(completion) - 150) / 500)
        rewards.append(max(0.0, (0.8 if correct else 0.0) - length_penalty))
    return rewards


REWARD_FUNCTIONS = {
    "r1": reward_r1,
    "r2": reward_r2,
    "r3": reward_r3,
    "r4": reward_r4,
}
