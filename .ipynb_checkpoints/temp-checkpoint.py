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
    
    
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0)

def query_llm(llm, prompts, verbose=False):
    outputs = llm.generate(prompts, sampling_params, use_tqdm=verbose)
    
    return [output.outputs[0].text for output in outputs]

def getOptionsString(options):
    res = ''
    for i in options:
        res += i + ' ' + options[i]
        res += '\n'
    return res

def getLetter(llm_response):
    # for i in llm_response.split('\n'):
    #     if(len(i) > 1 and i[0] != '#'):
    #         return i[0]
    
    for i in llm_response:
        if i.isupper():
            return i
    return '*'

def select_random_subset(f, subset_size=1000):
    """
    Select a random subset of specified size from the list f.

    Parameters:
    f (list): The list to select from.
    subset_size (int): The size of the subset to select.

    Returns:
    list: A random subset of the original list.
    """
    if subset_size > len(f):
        raise ValueError("Subset size is larger than the size of the input list")
    
    return random.sample(f, subset_size)

temp = pd.read_json(path_or_buf='/home/tgao/Github/llm/data_clean/questions/US/US_qbank.jsonl', lines=True)

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

n_samples = 10
idx = 0

res = []

bads = 0

while idx < n_samples:
    i = random.choice(range(len(data)))
    
    if(data['answer'][i] == data['model_letter'][i]):
        continue

    original_q = data['question'][i]
    options_str = getOptionsString(temp['options'][i])
    
    found = False
    
    all_qs = []
    for j in get_all_edits(original_q, 1):
        
        query = 'Question -\n\n' + j + '\nChoices -\n' + options_str
        query += '\nAs an extremely experienced and knowledgable medical professional answering this question accurately, the letter of the correct answer is '
        
        all_qs.append(query)
    
    if len(all_qs) > 1000:
        all_qs = select_random_subset(all_qs, 1000)
    
    print(len(all_qs))
    
    llm_ans = query_llm(llm, all_qs, verbose=True)
        
    for i in llm_ans:
        if(getLetter(i) != data['answer'][i]):
            print(i, data['answer'][i])
            print('=' * 30)
            
            res.append((i, data['answer'][i]))
            
            found = True
            break
        
    bads += 1
    
print(res)