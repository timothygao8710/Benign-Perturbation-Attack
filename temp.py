from vllm import LLM, SamplingParams
import pandas as pd
from string_functions import *
from llm_functions import *
from tqdm import tqdm
import json
import random

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

data = pd.read_csv('./llama_out.csv')

n_questions = 100 # how many questions to test
n_samples = 100 # how many samples to test, for each question

idx = 0

res = []

bads = 0

mp = {}

for iter in tqdm(range(n_samples)):
    i = random.choice(range(len(data)))
    
    if data['answer'][i] != data['model_letter'][i] or i in mp:
        iter -= 1
        continue

    mp[i] = {}

    original_q, options_str  = data['question'][i], getOptionsString(data['options'][i])
    mp[i]['question'] = original_q
    mp[i]['options'] = options_str
    mp[i]['correct_option'] = data['answer'][i]
    mp[i]['WA'] = []
    mp[i]['CA'] = []
    
    peturbed_qs = []

    changes = int(0.01 * (len(original_q) + len(options_str) + 162)) #round up?
    
    for j in sample_edits(original_q, changes, n_samples):
        
        query = 'Question -\n\n' + j + '\nChoices -\n' + options_str
        query += '\nAs an extremely experienced and knowledgable medical professional answering this question accurately, the letter of the correct answer is '
        
        # print(len(query) - len(j) - len(options_str))
        
        peturbed_qs.append(query)
    
    llm_ans = query_llm(llm, peturbed_qs, verbose=True)
            
    for j in range(len(llm_ans)):
        if(getLetter(llm_ans[j]) == data['answer'][i]):
            mp[i]['CA'].append(peturbed_qs[j])
        else:
            mp[i]['WA'].append(peturbed_qs[j])
        
    print(len(mp[i]['WA']) / (len(mp[i]['WA']) + len(mp[i]['CA'])))

out_file = open("out.json", "w") 
json.dump(mp, fp=out_file)
print(mp)