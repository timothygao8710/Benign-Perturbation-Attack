from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# Set up the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

toks = ["A", "B", "C", "D", "E", "F", "G", "Yes", "No"]
oks = [False for _ in range(len(toks))]

# Function to print all tokens and their characters
def print_all_tokens_and_characters():
    # Get the vocab dictionary
    vocab = tokenizer.get_vocab()
    
    # Sort tokens by their IDs
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
    
    # Print each token and its corresponding character
    for token, token_id in sorted_vocab:
        # print(f"Token ID: {token_id}, Token: '{token}'")
        for i in range(len(toks)):
            if(toks[i] == token):
                oks[i] = True
                print(f"Token ID: {token_id}, Token: '{token}'")

# Print all tokens and their corresponding characters
print_all_tokens_and_characters()

print(oks)