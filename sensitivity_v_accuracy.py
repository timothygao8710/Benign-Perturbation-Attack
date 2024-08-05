import random
import string
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import json
import random
import torch.nn.functional as F
import os
from semantic_uncertainty.calc_entropy import get_entropy_from_probabilities
from question_loader import *
import pickle
from utils import *

##### SETTINGS #####
cache_dir = '/tmp'
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
possible_outputs = ["A", "B", "C", "D", "E", "F", "G", "H"]
batch_size = 8
redownload = False
data_outpath = './data/edit_distance_v_accuracy_K_10'
######################

if redownload:
    model_cache_path = os.path.join(cache_dir, model_name)
    if os.path.exists(model_cache_path):
        os.rmdir(model_cache_path)


tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

def torch_to_numpy(torch_tensor):
    return torch_tensor.detach().cpu().numpy()

def get_next_token(prompt_batch, top_k=len(possible_outputs)):
    inputs = tokenizer(prompt_batch, padding = True, return_tensors="pt").to(model.device)

    allowed_tokens = tokenizer.convert_tokens_to_ids(possible_outputs)
    logits_bias = torch.full((len(prompt_batch), model.config.vocab_size), -float('inf')).to(model.device)
    logits_bias[:, allowed_tokens] = 0

    # print("Shape of input_ids:", inputs.input_ids.shape)
    # print("Shape of attention_mask:", inputs.attention_mask.shape)

    with torch.no_grad():
        outputs = model(**inputs)
        
        # Print shape of model output logits
        # print("Shape of model output logits:", outputs.logits.shape)
        
        next_token_logits = outputs.logits[:, -1, :] + logits_bias
        
        # Print shape of next_token_logits
        # print("Shape of next_token_logits:", next_token_logits.shape)
        
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Print shape of probs
        # print("Shape of probs:", probs.shape)
        
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
        
        # Print shapes of top_k results
        # print("Shape of top_k_indices:", top_k_indices.shape)
        # print("Shape of top_k_probs:", top_k_probs.shape)

    top_k_responses = [tokenizer.convert_ids_to_tokens(top_k_indices[i]) for i in range(len(prompt_batch))]
    return top_k_responses, torch_to_numpy(top_k_probs)

def perturb_input(input_text, edit_distance):
    """
    Randomly perturb the input text by the specified edit distance.
    """
    chars = list(input_text)
    for _ in range(edit_distance):
        operation = random.choice(['insert', 'delete', 'substitute'])
        if operation == 'insert':
            pos = random.randint(0, len(chars))
            chars.insert(pos, random.choice(string.ascii_letters))
        elif operation == 'delete' and chars:
            pos = random.randint(0, len(chars) - 1)
            chars.pop(pos)
        elif operation == 'substitute' and chars:
            pos = random.randint(0, len(chars) - 1)
            chars[pos] = random.choice(string.ascii_letters)
    return ''.join(chars)

def calculate_sensitivity(original_response, perturbed_responses):
    """
    Calculate sensitivity based on the original response and perturbed responses.
    """
    different_responses = sum(1 for resp in perturbed_responses if resp != original_response)
    return different_responses / len(perturbed_responses)

# Main execution
tot_questions = get_data_len()
# K = 10  # Edit distance for perturbation
K = [5, 10, 15, 20, 25]
n_samples = 5000
num_perturbations = 500  # Number of perturbations per input

res = load_data(data_outpath)
print("Loaded data from", data_outpath, "current rows:", len(res))
correct_count = 0

with tqdm(total=n_samples) as pbar:
    for iter in range(n_samples):
        row = random.randrange(tot_questions)
        
        cur_prompt = get_row_query(row)
            
        # Get original response
        original_response, original_probs = get_next_token([cur_prompt])
        original_ans = original_response[0][0]
        
        # Check accuracy
        cor_ans = get_correct_answer(row)
        is_correct = original_ans == cor_ans
        if is_correct:
            correct_count += 1
            
        # Calculate entropy
        entropy = get_entropy_from_probabilities(original_probs[0])
        
        # Update progress bar
        pbar.set_postfix({'Correct %': f'{(correct_count / (iter + 1)) * 100:.2f}%'})
        pbar.update(1)
        
        mp = {
                "row": row,
                "original_entropy": entropy,
                "is_correct": is_correct,
                "model_prob": original_probs[0].tolist(),
                "model_response": original_response[0]
            }
        
        for k in K:
            perturbed_prompts = [get_peturbed_row_query(row, k) for _ in range(num_perturbations)]
            
            # # Get responses for perturbed inputs --> memory error
            # perturbed_responses, peturbed_probs = get_next_token(perturbed_prompts)
            
            res_peturbs = [get_next_token([prompt]) for prompt in perturbed_prompts]
            perturbed_responses = [res_peturb[0] for res_peturb in res_peturbs]
            peturbed_probs = [res_peturb[1] for res_peturb in res_peturbs]

            tot_correct = 0
            tot_same = 0
            peturbed_entropies = []
            for i in range(len(perturbed_responses)):
                cur_ans = perturbed_responses[i][0][0]
                
                # print(peturbed_probs, original_ans)
                if cur_ans == original_ans:
                    tot_same += 1
                    
                if cur_ans == cor_ans:
                    tot_correct += 1
                    
                peturbed_entropies.append(get_entropy_from_probabilities(peturbed_probs[i]))
                    
            sensitivity_correct = tot_correct / num_perturbations
            sensitivity_same = tot_same / num_perturbations
            avg_entropy = sum(peturbed_entropies) / num_perturbations
            
            assert sensitivity_correct >= 0
            assert sensitivity_same >= 0
            
            mp[f"peturbed_entropy_{k}"] = avg_entropy
            mp[f"same_sensitivity_{k}"] = sensitivity_same
            mp[f"correct_sensitivity_{k}"] = sensitivity_correct
        
        print(mp)
        
        res.append(mp)
        
        if iter % 50 == 0:
            print("Iteration:", iter, "saved to", data_outpath)
            # Save results
            dump_data(res, data_outpath)

print(res)
dump_data(res, data_outpath)