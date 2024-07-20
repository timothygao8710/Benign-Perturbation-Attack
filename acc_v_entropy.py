from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import json
import random
import torch.nn.functional as F

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir = "/scratch"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")


def generate_text(prompts, max_length=100, top_k=4):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    allowed_tokens = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"])
    logits_bias = torch.full((len(prompts), tokenizer.vocab_size), -float('inf')).to(model.device)
    logits_bias[:, allowed_tokens] = 0
    
    results = []
    
    with torch.no_grad():
        for i in range(max_length):
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :] + logits_bias
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Get top K tokens and probabilities
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            
            results.append((top_k_indices, top_k_probs))
            
            # For the next iteration, use the most probable token as input
            next_input = top_k_indices[:, 0].unsqueeze(-1)
            inputs = tokenizer.build_inputs_for_generation(inputs, next_input)
            
            # Stop if all sequences have generated an end token
            if all(tokenizer.eos_token_id in seq for seq in inputs.input_ids):
                break
    
    # Convert results to readable format
    readable_results = []
    for batch_idx in range(len(prompts)):
        sequence_result = []
        for step in results:
            tokens = tokenizer.convert_ids_to_tokens(step[0][batch_idx])
            probs = step[1][batch_idx].tolist()
            sequence_result.append(list(zip(tokens, probs)))
        readable_results.append(sequence_result)
    
    return readable_results

# Usage:
prompts = ["Question 1: ...", "Question 2: ...", "Question 3: ..."]
results = generate_text(prompts)


### TODO: Finish finding corr bewteen acc and entropy. Sensitivity vs entropy. Ability to be quantized (inference speed, important for medical) vs entropy?
# In general, analyze relationship between sem entropy and things for medical setting.
# Then, prefix search (randomly), try to finid at least one that is non-negative accuracy delta
# Try to do chain of thought for "digging" based on how uncertian the model is