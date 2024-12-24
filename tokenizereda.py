import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer_path = "./SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
token_dict = tokenizer.get_vocab()
count = 0
for token, token_id in token_dict.items():
    count += 1
    print(token, token_id)
    if count > 10:
        break
