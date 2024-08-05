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
from torch.nn import functional as F
from utils import *

##### SETTINGS #####
cache_dir = '/tmp'
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
possible_outputs = ["A", "B", "C", "D", "E", "F", "G", "H"]
batch_size = 8
redownload = False
data_outpath = './data/noised_data'
######################

if redownload:
    model_cache_path = os.path.join(cache_dir, model_name)
    if os.path.exists(model_cache_path):
        os.rmdir(model_cache_path)


tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

orig_forward = model.base_model.embed_tokens.forward # For LLama

def torch_to_numpy(torch_tensor):
    return torch_tensor.detach().cpu().numpy()

def add_embed_noise(model, noise_alpha=5):
    def noised_embed(orig_embed, noise_alpha):
        def new_func(x):
            embed_init = orig_embed(x)
            dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
            mag_norm = noise_alpha/torch.sqrt(dims)
            return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
        return new_func

    model.base_model.embed_tokens.forward = noised_embed(orig_forward, noise_alpha)
    return model

def get_next_token(prompt_batch, top_k=len(possible_outputs)):
    inputs = tokenizer(prompt_batch, padding = True, return_tensors="pt").to(model.device)

    allowed_tokens = tokenizer.convert_tokens_to_ids(possible_outputs)
    logits_bias = torch.full((len(prompt_batch), model.config.vocab_size), -float('inf')).to(model.device)
    logits_bias[:, allowed_tokens] = 0
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :] + logits_bias
        probs = F.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
    top_k_responses = [tokenizer.convert_ids_to_tokens(top_k_indices[i]) for i in range(len(prompt_batch))]
    return top_k_responses, torch_to_numpy(top_k_probs)


tot_questions = get_data_len()
noise_alphas = [1,2,3,4,5,6,7,8,9,10]
n_samples = 1000
num_perturbations = 250

res = load_data(data_outpath)
correct_count = 0

with tqdm(total=n_samples) as pbar:
    for iter in range(n_samples):
        row = random.randrange(tot_questions)
        
        cur_prompt = get_row_query(row)
            
        model = add_embed_noise(model, 0)
        original_response, original_probs = get_next_token([cur_prompt])
        original_ans = original_response[0][0]
        
        cor_ans = get_correct_answer(row)
        is_correct = original_ans == cor_ans
        if is_correct:
            correct_count += 1
            
        entropy = get_entropy_from_probabilities(original_probs[0])
        
        pbar.set_postfix({'Correct %': f'{(correct_count / (iter + 1)) * 100:.2f}%'})
        pbar.update(1)
        
        mp = {
                "row": row,
                "original_entropy": entropy,
                "is_correct": is_correct,
                "model_prob": original_probs[0].tolist(),
                "model_response": original_response[0]
            }
        
        for k in noise_alphas:
            sensitivity_correct, sensitivity_same = 0, 0
            noised_entropies = []
            model = add_embed_noise(model, k)
            
            for trial in range(num_perturbations):

                response, probs = get_next_token([cur_prompt])
                ans = response[0][0]
                
                if ans == original_ans:
                    sensitivity_same += 1
                    
                if ans == cor_ans:
                    sensitivity_correct += 1
                    
                noised_entropies.append(get_entropy_from_probabilities(probs[0]))
                        
            sensitivity_correct = sensitivity_correct / num_perturbations
            sensitivity_same = sensitivity_same / num_perturbations
            avg_entropy = sum(noised_entropies) / num_perturbations
            
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