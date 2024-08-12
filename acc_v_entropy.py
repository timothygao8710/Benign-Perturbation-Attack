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
from LLM import get_next_token_fast
import numpy as np

##### SETTINGS #####
cache_dir = '/tmp'
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
possible_outputs = ["A", "B", "C", "D", "E", "F", "G", "H"]
batch_size = 8
redownload = False
data_outpath = './data/all_entropies'
######################

if redownload:
    model_cache_path = os.path.join(cache_dir, model_name)
    if os.path.exists(model_cache_path):
        os.rmdir(model_cache_path)

# Turning into classifier
# Use longer responses, see which one is better --> logit bias or entailment w/out logit bias
# average log prob over sentence nromalized

# Semnatic entropy, [Prob of A, Prob of B...], [Diff of A, Diff of B...], embedding layer probs

# 73% - something in the prompt that makes model over confident
# Feeding in semantically similar prompts into model

# Sentence normalized average semantic entropy over all tokens...???

# One metric: How much performance improvement by ignoring low confidence samples

### TODO: Finish finding corr bewteen acc and entropy. Sensitivity vs entropy. Ability to be quantized (inference speed, important for medical) vs entropy?
## TODO: ADD AUROC for analysis!!! <<--- nvm, just add desntiy plot plz
# In general, analyze relationship between sem entropy and things for medical setting.
# Then, prefix search (randomly), try to finid at least one that is non-negative accuracy delta
# Try to do chain of thought for "digging" based on how uncertian the model is

# How to make LLM better at estimating how confident it is in its answer
# Intuitively, can we actually do better thanusing embedding layer? Because there's the idea might just be confidently wrong... if its unsure that would be reflected in 

# semantic entropy over all perturbations
# semantically similar prompts, how much does the model's confidence change?
# thoguht experiemtns?

# try different perrubation methods (levels (characters, word, setnence/semantic similar))

# TODO: Find pattern in hallucinated vs non-hullucated prmopts. Try to find why, to decerase confident wrorng answers
# TODO: Instead of using edit distance, can we add noise directly to the embedding layer? (semantic noise)
# Look at attention weights, see if there's a pattern in the attention weights that are wrong

# features: semantic entropy, perturbations, 

# write down time for experiemetn

def arr_to_prob_freq(arr):
    mp = {}
    for i in arr:
        if i not in mp:
            mp[i] = 0
        mp[i] += 1
    return sorted([(i, mp[i] / len(arr)) for i in mp.keys()], key=lambda x: -x[1])
    

# if __name__ == '__main__':
#     prompts = ["Question 1 test asdf lmao yeet: ...", "Question 2: ...", "Say the letter G"]
#     results, probs = get_next_token(prompts)

#     print(results)
#     print(probs)

#     print([get_entropy_from_probabilities(i) for i in probs])

tot_questions = 1000
# tot_questions = get_data_len()
# n_samples = 2000
print(tot_questions)

res = []
correct_count = 0
n_shuffles = 50
temp = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6, 'H' : 7}

with tqdm(total=tot_questions) as pbar:
    for row in range(tot_questions):
        cor_ans = get_correct_answer(row)
        tot_sem_entropy = 0
        all_responses = []
        all_entropies = []
        
        for _ in range(n_shuffles):
            cur_prompt, shuffled = get_shuffled_row_query(row)
            cur_prompt = [cur_prompt]

            response, probs = get_next_token_fast(cur_prompt)
            
            # print(response, probs)
            # print("SDF", sum(probs))
            
            entropy = get_entropy_from_probabilities(probs)
            tot_sem_entropy += entropy

            model_ans = response[0][0]
            model_ans = shuffled[temp[model_ans]]

            all_responses.append(model_ans)
            all_entropies.append(entropy)

        print(all_responses)
        
        answer_avg_entropy = sum(np.where(all_responses == cor_ans, all_entropies, 0)) / n_shuffles
        avg_entropy = sum(all_entropies) / n_shuffles

        model_ans = arr_to_prob_freq(all_responses)[0][0]
        correct_count += 1 if model_ans == cor_ans else 0
        # Update progress bar with the current percentage of correct answers
        pbar.set_postfix({'Correct %': f'{(correct_count / (row + 1)) * 100:.2f}%'})
        pbar.update(1)

        # print(row, is_correct, entropy, model_ans, cor_ans, probs[0])
        res.append({
                "row": row,
                "avg_entropy": avg_entropy,
                "answer_avg_entropy": answer_avg_entropy,
                "is_correct": model_ans == cor_ans,
                "model_prob":arr_to_prob_freq(all_responses),
                "model_response": model_ans,
            })

dump_data(res, data_outpath)