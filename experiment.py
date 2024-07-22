import random
import string
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from semantic_uncertainty.calc_entropy import get_entropy_from_probabilities
from question_loader import *

# Settings
cache_dir = '/tmp'
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
possible_outputs = ["A", "B", "C", "D", "E", "F", "G", "H"]
batch_size = 8
n_samples = 200
n_prefix_attempts = int(1e6)
max_prefix_length = 20
baseline_value = 0.63  # 63% baseline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

def generate_random_prefix(max_length):
    length = random.randint(5, max_length)
    return ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation + ' ') for _ in range(length))

def get_next_token(prompt_batch, top_k=len(possible_outputs)):
    inputs = tokenizer(prompt_batch, padding=True, return_tensors="pt").to(model.device)
    allowed_tokens = tokenizer.convert_tokens_to_ids(possible_outputs)
    logits_bias = torch.full((len(prompt_batch), model.config.vocab_size), -float('inf')).to(model.device)
    logits_bias[:, allowed_tokens] = 0

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :] + logits_bias
        probs = F.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)

    top_k_responses = [tokenizer.convert_ids_to_tokens(top_k_indices[i]) for i in range(len(prompt_batch))]
    return top_k_responses, top_k_probs.cpu().numpy()

def evaluate_accuracy(prefix=""):
    tot_questions = get_data_len()
    correct_count = 0

    for _ in range(n_samples):
        row = random.randrange(tot_questions)
        cur_prompt = [get_row_query(row) + '\n\n' + prefix]
        response, probs = get_next_token(cur_prompt)

        model_ans = response[0][0]
        cor_ans = get_correct_answer(row)

        if model_ans == cor_ans:
            correct_count += 1

    return correct_count / n_samples

def find_better_prefixes():
    # baseline_accuracy = evaluate_accuracy()  # Baseline accuracy without prefix
    baseline_accuracy = baseline_value
    print(f"Baseline accuracy: {baseline_accuracy:.2%}")

    better_prefixes = []

    for _ in tqdm(range(n_prefix_attempts)):
        current_prefix = generate_random_prefix(max_prefix_length)
        current_accuracy = evaluate_accuracy(current_prefix)

        if current_accuracy > baseline_value:
            better_prefixes.append((current_prefix, current_accuracy))
            print(f"Prefix found above baseline: '{current_prefix}' with accuracy: {current_accuracy:.2%}")

    return better_prefixes, baseline_accuracy

if __name__ == '__main__':
    better_prefixes, baseline_accuracy = find_better_prefixes()
    
    print(f"\nBaseline accuracy: {baseline_accuracy:.2%}")
    print(f"Number of prefixes found above {baseline_value:.2%} baseline: {len(better_prefixes)}")

    if better_prefixes:
        print("\nPrefixes that performed better than the baseline:")
        for prefix, accuracy in better_prefixes:
            print(f"Prefix: '{prefix}', Accuracy: {accuracy:.2%}")

        # Sort prefixes by accuracy and get the best one
        best_prefix, best_accuracy = max(better_prefixes, key=lambda x: x[1])
        print(f"\nBest prefix found: '{best_prefix}'")
        print(f"Best accuracy: {best_accuracy:.2%}")

        # Save results
        results = {
            "baseline_accuracy": baseline_accuracy,
            "baseline_value": baseline_value,
            "better_prefixes": [{"prefix": p, "accuracy": a} for p, a in better_prefixes],
            "best_prefix": best_prefix,
            "best_accuracy": best_accuracy
        }
    else:
        print("\nNo prefixes found that performed better than the baseline.")
        results = {
            "baseline_accuracy": baseline_accuracy,
            "baseline_value": baseline_value,
            "better_prefixes": []
        }
    
    with open('prefix_search_results.json', 'w') as file:
        json.dump(results, file, indent=2)

    print("Results saved to 'prefix_search_results.json'")