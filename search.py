from vllm import LLM, SamplingParams
import pandas as pd

# jsonObj = pd.read_json(path_or_buf='/home/tgao/Github/llm/data_clean/questions/US/train.jsonl', lines=True)
jsonObj = pd.read_json(path_or_buf='/home/tgao/Github/llm/data_clean/questions/US/US_qbank.jsonl', lines=True)

# llm = LLM(model="openlm-research/open_llama_13b")

print(jsonObj.columns)


