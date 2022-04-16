def clean_model_name(model_name):
    return model_name.replace('/', '-')


data_headers = ['premise', 'hypothesis', "entailment", 'reasoning_type', 'function_name', 'g_id']
results_headers = ['premise', 'hypothesis', "entailment", 'reasoning_type', 'function_name', 'g_id', 'answer', 'correct']

def append_generator_to_tsv(tsv_writer, gen):
    for (p, h), e, (r_type, fn_name, id_) in gen:
        #print(*[p, h, e, r_type, fn_name, id_])
        tsv_writer.writerow([p, h, e, r_type, fn_name, id_])
        

def transform_gpt3_model_name(gpt3_model_name):
    if gpt3_model_name == "davinci":
        return "text-davinci-002"
    else:
        return f"text-{gpt3_model_name}-001"

few_shot_prompt_template = '''A soccer game with multiple males playing.
Question: Some men are playing a sport. True, false, or neither?
Answer: True

A statue at a museum that no seems to be looking at.
Question: Tons of people are gathered around the statue. True, false, or neither?
Answer: False

A woman with a green headscarf, blue shirt and a very big grin.
Question: The woman is young. True, false, or neither?
Answer: Neither

{premise}
Question: {hypothesis}
Answer:'''