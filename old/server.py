import ray
ray.init(ignore_reinit_error=True, num_cpus=4)
from vllm import LLM, SamplingParams
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", tensor_parallel_size=8)