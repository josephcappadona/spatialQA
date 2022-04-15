import openai, os

openai.organization = "org-IISEC8WXjZjP5AMr5pepYRVs"
openai.api_key = os.getenv("OPENAI_API_KEY")

gpt3_prompt_template = A soccer game with multiple males playing.
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
Answer:

prompt1 = gpt3_prompt_template.format(premise="John is running.",
                                      hypothesis="John is in motion")

prompt2 = gpt3_prompt_template.format(premise="John is skipping class.",
                                      hypothesis="John is in motion")

print(openai.Completion.create(
  engine="text-davinci-002",
  prompt=[prompt1, prompt2],
  temperature=1,
  max_tokens=5
))