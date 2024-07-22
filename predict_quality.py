import json
import ast

import torch
import torch.nn as nn
from torch.nn import functional as F

file_path = 'acc_v_entropy.json'

def load_data(file_path):
    with open(file_path, 'r') as file:
        data_str = json.load(file)
        data = ast.literal_eval(data_str)
    
    features, output = [], []

    for row in data:
        output.append(row["is_correct"])

        cur_features = [row["entropy"]]
        cur_features.extend(row["model_prob"])
        diffs = [row["model_prob"][i-1] - row["model_prob"][i] for i in range(1, len(row["model_prob"]))]
        cur_features.extend(diffs)
        
        features.append(cur_features)

    return features, output



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
