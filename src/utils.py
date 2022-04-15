def clean_model_name(model_name):
    return model_name.replace('/', '-')


data_headers = ['premise', 'hypothesis', "entailment", 'reasoning_type', 'function_name', 'g_id']
results_headers = ['premise', 'hypothesis', "entailment", 'reasoning_type', 'function_name', 'g_id', 'answer', 'correct']

def append_to_tsv(tsv_writer, gen):
    for (p, h), e, (r_type, fn_name, id_) in gen:
        print(*[p, h, e, r_type, fn_name, id_])
        tsv_writer.writerow([p, h, e, r_type, fn_name, id_])
        

def transform_gpt3_model_name(gpt3_model_name):
    if gpt3_model_name == "davinci":
        return "text-davinci-002"
    else:
        return f"text-{gpt3_model_name}-001"

gpt3_prompt_template = '''A soccer game with multiple males playing.
Question: Some men are playing a sport. True, false, or neither?
Answer: True

A statue at a museum that no seems to be looking at.
Question: Tons of people are gathered around the statue. True, false, or neither?
Answer: False

A woman with a green headscarf, blue shirt and a very big grin.
Question: The woman is young. True, false, or neither?
Answer: Neither

A man playing an electric guitar on stage.
Question: A man is performing for cash. True, false, or neither?
Answer: Neither

A blond-haired doctor is looking through new medical manuals.
Question: A doctor is looking at a book. True, false, or neither?
Answer: True

A young family enjoys feeling ocean waves lap at their feet.
Question: A family is out at a restaurant. True, false, or neither?
Answer: False

{premise}
Question: {hypothesis}
Answer:'''