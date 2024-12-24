import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from createConstraints import createFullConstraint
from constrainedInference import constrained_inference
import os


class ConstrainedDecoder:
    def __init__(
        self,
        model_name: str,
        constraints_dict: dict,
        device: str,
        window_size=float("inf"),
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.constraint = createFullConstraint(self.tokenizer, constraints_dict)
        self.window_size = window_size
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemError("CUDA is not available on this machine.")
        if device == "mps" and torch.backends.mps.is_available():
            raise SystemError("MPS is not available on this machine.")
        if device != "cuda" and device != "cpu" and device != "mps":
            raise ValueError("Invalid device type.")
        self.device = device
        model = model.to(device)

    def from_path(self, model_path, constraints_dict, device, window_size):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model path does not exist.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.constraint = createFullConstraint(self.tokenizer, constraints_dict)
        self.window_size = window_size
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemError("CUDA is not available on this machine.")
        if device == "mps" and torch.backends.mps.is_available():
            raise SystemError("MPS is not available on this machine.")
        if device != "cuda" and device != "cpu" and device != "mps":
            raise ValueError("Invalid device type.")
        self.device = device
        model = model.to(device)

    def decode(self, prompt, max_new_toks=50):
        return constrained_inference(
            self.model,
            prompt,
            self.tokenizer,
            self.constraint,
            device=self.device,
            max_new_toks=max_new_toks,
            window_size=self.window_size,
        )
