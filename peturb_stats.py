from vllm import LLM, SamplingParams
import pandas as pd
from string_functions import *
from llm_functions import *
from tqdm import tqdm
import json
import random

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

data = pd.read_csv('./llama_out.csv')

n_questions = 500 # how many questions to test
n_samples = 200 # how many samples to test, for each question

res_mp = {}

percentage_bins = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 0.75, 1]

for percent in percentage_bins:

    all_queries = []
    llm_input = []

    print('START', percent, '=' * 50)

    mp = {}
    iter = 0
    with tqdm(total=n_samples) as pbar:
        while iter < n_samples:
            i = random.choice(range(len(data)))
            
            if data['answer'][i] != data['model_letter'][i] or i in mp:
                continue
            iter += 1

            mp[i] = {}

            original_q, options_str  = data['question'][i], getOptionsString(data['options'][i])
            mp[i]['question'] = original_q
            mp[i]['options'] = options_str
            mp[i]['correct_option'] = data['answer'][i]
            mp[i]['WA'] = 0
            mp[i]['CA'] = 0
            
            peturbed_qs = []

            changes = int(percent * (len(original_q) + len(options_str) + 162)) #round up?
            
            for j in sample_edits(original_q, changes, n_samples):
                
                query = 'Question -\n\n' + j + '\nChoices -\n' + options_str
                query += '\nAs an extremely experienced and knowledgable medical professional answering this question accurately, the letter of the correct answer is '
                
                # print(len(query) - len(j) - len(options_str))
                
                all_queries.append([i, data['answer'][i], query])
                llm_input.append(query)
                        
    llm_ans = query_llm(llm, llm_input, verbose=True)

    for i in range(len(llm_ans)):
        idx = all_queries[i][0]
        cor_ans = all_queries[i][1]
        cur_q = all_queries[i][2]
        
        if(getLetter(llm_ans[i]) == cor_ans):
            mp[idx]['CA'] += 1
        else:
            mp[idx]['WA'] += 1
        
        # if (len(mp[idx]['WA']) + len(mp[idx]['CA'])) == n_samples:
        #     print(len(mp[idx]['WA']) / n_samples)
            
    res_mp[percent] = mp
    
    print(mp)
    
    cur_out_file = open(str(percent) + " out_percentage.json", "w") 
    json.dump(mp, fp=cur_out_file)
    
    print('END', percent, '=' * 50)

out_file = open("out_percentages.json", "w") 
json.dump(res_mp, fp=out_file)
print(res_mp)
