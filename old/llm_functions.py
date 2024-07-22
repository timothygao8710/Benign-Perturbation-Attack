from vllm import LLM, SamplingParams
from helper import *

sampling_params = SamplingParams(temperature=0)

def query_llm(llm, prompts, verbose=False):
    outputs = llm.generate(prompts, sampling_params, use_tqdm=verbose)
    
    return [output.outputs[0].text for output in outputs]

def getLetter(llm_response):
    # for i in llm_response.split('\n'):
    #     if(len(i) > 1 and i[0] != '#'):
    #         return i[0]
    
    for i in llm_response:
        if i.isupper():
            return i
    return '*'

def getQuery(question, options_str):
    query = 'Question -\n\n' + question + '\nChoices -\n' + options_str
    query += '\nAs an extremely experienced and knowledgable medical professional answering this question accurately, the letter of the correct answer is '
    return query
