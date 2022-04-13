import os
import openai
openai.organization = "org-IISEC8WXjZjP5AMr5pepYRVs"
openai.api_key = "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"


prompt = '''John is thinking.
Question: John is in motion. True, false, or neither?
Answer: Neither

John is standing.
Question: John is in motion. True, false, or neither?
Answer: False

John is skipping.
Question: John is in motion. True, false, or neither?
Answer: True

John is running.
Question: John is in motion. True, false, or neither?
Answer:'''

print(openai.Completion.create(
  engine="text-ada-001",
  prompt=prompt,
  temperature=0,
  max_tokens=1
))