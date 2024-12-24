from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

checkpoint = "HuggingFaceTB/SmolLM-135M"
device = "mps"

model_path = "SmolLM-135M"

if not os.path.exists(model_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
print(tokenizer.get_vocab()["I"])
print(tokenizer.get_vocab()["love"])
