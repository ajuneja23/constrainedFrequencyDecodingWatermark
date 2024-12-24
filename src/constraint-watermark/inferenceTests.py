from constrainedInference import constrained_inference
from createConstraints import createFullConstraint
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def fullInfer(model, tokenizer, prompt, constraint_dict, window_size):
    complete_dict = createFullConstraint(tokenizer, constraint_dict)
    return constrained_inference(
        model, prompt, tokenizer, complete_dict, device="cpu", window_size=window_size
    )


"""
model_path = "./SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = "mps"
constraint_dict = {"I": 0, "love": 1, "trucks": 2}
window_size = 20
prompt = "Hey there, how are you doing today?"

output_file = "inference_results.json"

# Store results in a list
results = []


# Perform inference 100 times
for i in range(100):
    result = fullInfer(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        constraint_dict=constraint_dict,
        window_size=window_size,
    )
    results.append({"iteration": i + 1, "result": result})

# Write results to a file in JSON format
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
"""
