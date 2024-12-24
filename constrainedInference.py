import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def constrained_inference(
    model,
    prompt,
    tokenizer,
    constrain_dict,  # contains allowed freqs per window in generated response
    device="cpu",
    max_new_toks=50,
    window_size=100,
):
    with torch.no_grad():
        model = model.to(device)
        tokens = tokenizer(prompt, return_tensors="pt")
        new_toks = 0
        cur_tok = None
        while new_toks <= max_new_toks and cur_tok != tokenizer.eos_token_id:
            logits = oneTokenInfer(model, tokens, constrain_dict, device)
            cur_tok = torch.multinomial(logits, 1).item()
            tokens["input_ids"] = torch.cat(
                [tokens["input_ids"], torch.tensor([[cur_tok]]).to(device)], dim=1
            )
            tokens["attention_mask"] = torch.cat(
                [tokens["attention_mask"], torch.tensor([[1]]).to(device)], dim=1
            )
            new_toks += 1
            constrain_dict[cur_tok] -= 1
            if new_toks >= window_size:
                constrain_dict[tokens["input_ids"][0][-window_size]] += 1
        return tokenizer.decode(tokens["input_ids"][0])


def oneTokenInfer(
    model,
    tokens,
    constrain_dict,
    device="cpu",
):
    with torch.no_grad():
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        logits = logits[:, -1, :]
        logits = logits.squeeze()
        for token, freq in constrain_dict.items():
            if freq == 0:
                logits[token] = 0
        if torch.sum(logits) == 0:
            raise ValueError("No tokens available")
        logits = logits / torch.sum(logits)
        return logits
