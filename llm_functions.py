from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def query(llm, prompts, verbose=False):
    outputs = llm.generate(prompts, sampling_params)
    
    return [output.outputs[0].text for output in outputs]
        