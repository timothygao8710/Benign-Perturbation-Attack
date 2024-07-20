from vllm import LLM, SamplingParams
import pandas as pd
from llm_functions import query_llm
from tqdm import tqdm

# llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

data = pd.read_csv('./llama_out.csv')


WA = "A 5 year-old-boy with a history of severe allergies and recurrent sinusitis presents with foul-smelling, fatty diarrhea. He is at the 50th percentile for height and weight. The boy's mother reports that he has had several such episodes of diarrhea over the years. He does not have any known history of fungal infections or severe viral infections. Which of the following is the most likely underlying cause of this boy's presentation?"

CA = "A 68-year-old woman comes to the physician with dysphagia and halitosis for several months. She feels food sticking to her throat immediately after swallowing. Occasionally, she regurgitates undigested food hours after eating. She has no history of any serious illness and takes no medications. Her vital signs are within normal limits. Physical examination including the oral cavity, throat, and neck shows no abnormalities. Which of the following is the most appropriate diagnostic study at this time?"

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
    if(data['question'][i][:100] != WA[:100] and data['question'][i][:100] != CA[:100]):
        continue
    
    print(data['options'][i])
    
    query = 'Question -\n\n' + data['question'][i] + '\nChoices -\n' + getOptionsString(data['options'][i]) 
    
    query += '\nAs an extremely experienced and knowledgable medical professional answering this question accurately, the letter of the correct answer is '
    
    answer = data['answer'][i]
    
    print(query)
    print(answer)
    
    prompts.append(query)
    answers.append(answer)


print(prompts, answers)
llm_ans = query_llm(llm, prompts, verbose=True)

print(llm_ans)

# for j in range(n):
#     if(getLetter(llm_ans[j]) == answers[j]):
#         n_correct += 1
    
#     model_ans.append(llm_ans[j])
#     model_letter.append(getLetter(llm_ans[j]))

# # print(answers)
# # print(model_letter)
# print(n_correct / n)

# data['model_letter'] = model_letter
# data['model_response'] = model_ans

# data.to_csv('test.csv', index=False)