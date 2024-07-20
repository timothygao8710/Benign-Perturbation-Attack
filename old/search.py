from vllm import LLM, SamplingParams
import pandas as pd
from llm_functions import query_llm
from tqdm import tqdm
import random

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

data = pd.read_csv('/home/tgao/Github/llm/llama_out.csv')

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def get_all_edits(word, n=1):
    if n == 1:
        return edits1(word)
    
    last = get_all_edits(word, n-1)

    res = set()
    for i in last:
        cur = edits1(i)
        for j in cur:
            res.add(j)
    return res

n_samples = 1000
idx = 0


while idx < n_samples:
    i = random.choice(range(len(data)))
    
    if(data['answer'][i] == data['model_letter'][i]):
        continue
    
    