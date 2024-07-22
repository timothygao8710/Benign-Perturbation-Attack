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
data_outpath = './data/medqa_llama'
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


# if __name__ == '__main__':
#     prompts = ["Question 1 test asdf lmao yeet: ...", "Question 2: ...", "Say the letter G"]
#     results, probs = get_next_token(prompts)

#     print(results)
#     print(probs)

#     print([get_entropy_from_probabilities(i) for i in probs])


### TODO: Finish finding corr bewteen acc and entropy. Sensitivity vs entropy. Ability to be quantized (inference speed, important for medical) vs entropy?
## TODO: ADD AUROC for analysis!!! <<--- nvm, just add desntiy plot plz
# In general, analyze relationship between sem entropy and things for medical setting.
# Then, prefix search (randomly), try to finid at least one that is non-negative accuracy delta
# Try to do chain of thought for "digging" based on how uncertian the model is

tot_questions = get_data_len()
n_samples = 2000
print(tot_questions)

res = []
correct_count = 0

with tqdm(total=n_samples) as pbar:
    for _ in range(n_samples):
        row = random.randrange(tot_questions)
        cur_prompt = [get_row_query(row)]

        response, probs = get_next_token(cur_prompt)

        entropy = get_entropy_from_probabilities(probs[0])

        model_ans = response[0][0]
        model_ans_prob = probs[0][0]
        cor_ans = get_correct_answer(row)

        # Check if the model's answer is correct
        is_correct = model_ans == cor_ans
        if is_correct:
            correct_count += 1

        # Update progress bar with the current percentage of correct answers
        pbar.set_postfix({'Correct %': f'{(correct_count / (_ + 1)) * 100:.2f}%'})
        pbar.update(1)

        # print(row, is_correct, entropy, model_ans, cor_ans, probs[0])
        res.append({
                "row": row,
                "entropy": entropy,
                "is_correct": is_correct
            })

with open('acc_v_entropy.json', 'w') as file:
    json.dump(str(res), file, indent=2)

file.close()