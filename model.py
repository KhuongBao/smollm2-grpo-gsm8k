import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import CHECKPOINT, DEVICE, CHAT_CONFIG


def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model_from_path(path):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer


def save_model(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def chat(model, tokenizer, question):
    prompt = f"Question: {question}\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        **inputs,
        **CHAT_CONFIG,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_length = inputs.input_ids.shape[-1]
    response = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

    return response
