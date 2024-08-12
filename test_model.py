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

cur_prompt = [
    
'''
Question -

A six-year-old male is seen in your ED due to new-onset fever, headache, lethargy, and the rash seen in Figure A. Measurement of the skin lesions reveals them to be 0.7cm in size on average, non-palpable, and non-blanching. What is the appropriate term for this skin finding?

Choices -

A Petechiae
B Purpura
C Ecchymosis
D Plaques
E Patches

As an extremely experienced and knowledgeable medical professional answering this question accurately, the letter of the correct answer is
'''
]

response, probs = get_next_token(cur_prompt)