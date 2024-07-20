from vllm import LLM, SamplingParams
import pandas as pd
from llm_functions import query_llm
from tqdm import tqdm

# llm = LLM(model="openlm-research/open_llama_13b")
# llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
# llm = LLM(model="mistral-community/Mixtral-8x22B-v0.1")
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

# jsonObj = pd.read_json(path_or_buf='/home/tgao/Github/llm/data_clean/questions/US/train.jsonl', lines=True)
data = pd.read_json(path_or_buf='/home/tgao/Github/llm/data_clean/questions/US/US_qbank.jsonl', lines=True)
print(data.head())

instruction = """
You are a licensed medical professional. Using your expertise, select one out of the avaliable multiple-choice
options to the following question. \n\n
"""

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
        

n = len(data)
n_correct = 0

model_ans = []
model_letter = []

prompts = []
answers = []

for i in range(n):
    query = 'Question -\n\n' + data['question'][i] + '\nChoices -\n' + getOptionsString(data['options'][i]) 
    
    
    query += '\nAs an extremely experienced and knowledgable medical professional answering this question accurately, the letter of the correct answer is '
    
    answer = data['answer'][i]
    
    # print(query)
    # print(answer)
    
    prompts.append(query)
    answers.append(answer)
        
llm_ans = query_llm(llm, prompts, verbose=True)

print(llm_ans)

for j in range(n):
    if(getLetter(llm_ans[j]) == answers[j]):
        n_correct += 1
    
    model_ans.append(llm_ans[j])
    model_letter.append(getLetter(llm_ans[j]))

# print(answers)
# print(model_letter)
print(n_correct / n)

data['model_letter'] = model_letter
data['model_response'] = model_ans

data.to_csv('test.csv', index=False)