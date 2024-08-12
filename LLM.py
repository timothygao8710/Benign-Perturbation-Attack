import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np

##### SETTINGS #####
cache_dir = '/tmp'
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
possible_outputs = ["A", "B", "C", "D", "E", "F", "G", "H"]
# possible_outputs = ["Yes", "No"]
batch_size = 8
redownload = False
data_outpath = './data/all_entropies'
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

def get_next_token_fast(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    allowed_tokens = tokenizer.convert_tokens_to_ids(possible_outputs)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        filtered_logits = next_token_logits[allowed_tokens]
        probs = F.softmax(filtered_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        return np.take(possible_outputs, torch_to_numpy(sorted_indices)), torch_to_numpy(sorted_probs)

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

def generate(prompt, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate until EOS token
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.001,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

if __name__ == "__main__":
    # all_tokens = tokenizer.get_vocab()
    # print("Yes" in all_tokens)
    # print("No" in all_tokens)
    # print("True" in all_tokens)
    # print("False" in all_tokens)

    prompt = '''Question -

    A 4670-g (10-lb 5-oz) male newborn is delivered at term to a 26-year-old woman after prolonged labor. Apgar scores are 9 and 9 at 1 and 5 minutes. Examination in the delivery room shows swelling, tenderness, and crepitus over the left clavicle. There is decreased movement of the left upper extremity. Movement of the hands and wrists are normal. A grasping reflex is normal in both hands. An asymmetric Moro reflex is present. The remainder of the examination shows no abnormalities and an anteroposterior x-ray confirms the diagnosis. Which of the following is the most appropriate next step in management?

    Choices -

    A Nerve conduction study
    B MRI of the clavicle

    As an extremely experienced and knowledgeable medical professional, which response is more appropriate? Respond with A or B.
    
    Solution - 
    '''

    # responses, probs = get_next_token_fast(prompt)
    # print(responses, probs)
    # responses, probs = get_next_token([prompt])
    # print(responses, probs)
    # print("Done!")
    
    print(generate(prompt))

