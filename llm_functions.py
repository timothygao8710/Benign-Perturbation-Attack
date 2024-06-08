from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0)

def query_llm(llm, prompts, verbose=False):
    outputs = llm.generate(prompts, sampling_params, use_tqdm=verbose)
    
    return [output.outputs[0].text for output in outputs]
