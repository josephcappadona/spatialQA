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


unifiedqa_MC_template = '''Some men are playing a sport.
(A) True (B) False (C) Neither
A soccer game with multiple males playing.
True

Tons of people are gathered around the statue.
(A) True (B) False (C) Neither
A statue at a museum that no seems to be looking at.
False

The woman is young.
(A) True (B) False (C) Neither
A woman with a green headscarf, blue shirt and a very big grin.
Neither

{hypothesis}
(A) True (B) False (C) Neither
{premise}
'''

gpt_prompt_template = '''A soccer game with multiple males playing.
Question: Some men are playing a sport. True, false, or neither?
Answer: True

A statue at a museum that no seems to be looking at.
Question: Tons of people are gathered around the statue. True, false, or neither?
Answer: False

A woman with a green headscarf, blue shirt and a very big grin.
Question: The woman is young. True, false, or neither?
Answer: Neither

{premise}
Question: {hypothesis} True, false, or neither?
Answer:'''

def tfn_decode(y):
    try:
        y = y.strip().split()[0].lower()
    except Exception:
        return "None"
    if y == "true":
        return "entailment"
    elif y == "false":
        return "contradiction"
    elif y == "neither":
        return "neutral"
    else:
        return "None"
