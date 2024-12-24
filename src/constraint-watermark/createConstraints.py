import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def createFullConstraint(tokenizer, constraint_dict):
    """
    constraint_dict: dictionary of constraints on specific tokens.  This function
      will tokenize your constraints and map to ids, while also setting constraints
      for the tokens you did not input to infinity. For example you could input to
      this function
      {
        "I": 0,
        "love": 1,
        "trucks": 2
      }
      And the output would be a dictionary of vocab_length elements, with vocab_length-3
      elements set to infinity and the rest set to the values you input. If any keys in constraint_dict
      do not map to a token then they are ignored.
    """
    complete_dict = {}
    token_dict = tokenizer.get_vocab().items()
    for token, token_id in token_dict:
        complete_dict[token_id] = float("inf")
    for key, maxFreq in constraint_dict.items():
        if key in tokenizer.get_vocab():
            complete_dict[tokenizer.get_vocab()[key]] = maxFreq
    return complete_dict
