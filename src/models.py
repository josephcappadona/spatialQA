from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import openai
from utils import few_shot_prompt_template, transform_gpt3_model_name



def get_model(model_name, device):
    
    if 'unifiedqa' in model_name:
        return get_unifiedqa(model_name, device)

    elif 't5' in model_name:
        return get_t5(model_name, device)

    elif 'mnli' in model_name.lower():
        return get_mnli(model_name, device)

    elif ('ada' in model_name or 
            'babbage' in model_name or 
            'curie' in model_name or 
            'davinci' in model_name):
        return get_gpt3(model_name)


def get_unifiedqa(model_name, device):

    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    def t5_encode(batch_x):
        transform_x = lambda x: few_shot_prompt_template.format(premise=x[0], hypothesis=x[1])
        transformed = list(map(transform_x, batch_x))
        return tokenizer(transformed, return_tensors="pt", padding=True)
    
    def t5_run_model(inputs):
        return model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            do_sample=False,
        )
    
    def t5_decode(outputs):
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return t5_run_model, t5_encode, t5_decode


def get_t5(model_name, device):

    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    def t5_encode(batch_x):
        transform_x = lambda x: f"mnli hypothesis: {x[1]} premise: {x[0]}"
        transformed = list(map(transform_x, batch_x))
        return tokenizer(transformed, return_tensors="pt", padding=True)
    
    def t5_run_model(inputs):
        return model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            do_sample=False,
        )
    
    def t5_decode(outputs):
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return t5_run_model, t5_encode, t5_decode
    

openai.organization = "org-IISEC8WXjZjP5AMr5pepYRVs"
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_gpt3(model_name):

    api_model_name = transform_gpt3_model_name(model_name)

    def gpt3_encode(batch_x):
        transform_x = lambda x: few_shot_prompt_template.format(premise=x[0], hypothesis=x[1])
        return [transform_x(x) for x in batch_x]
    
    def gpt3_run_model(inputs):
        response = openai.Completion.create(
            engine=api_model_name,
            prompt=inputs,
            temperature=0,
            max_tokens=1
        )
        return [d["text"] for d in response["choices"]]

    def gpt3_decode(outputs):
        def transform_y(y):
            y = y.strip().lower()
            if y == "true":
                return "entailment"
            elif y == "false":
                return "contradiction"
            elif y == "neither":
                return "neutral"
            else:
                return ""
        return [transform_y(y) for y in outputs]
    
    return gpt3_run_model, gpt3_encode, gpt3_decode
    

def get_mnli(model_name, device):
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def mnli_encode(batch_x):
        return tokenizer.batch_encode_plus(batch_x,
                                            truncation="only_first",
                                            return_tensors="pt",
                                            padding=True)
    
    def mnli_run_model(inputs):
        return model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )[0]
    
    def mnli_decode(outputs):

        answers = torch.argmax(outputs, axis=1).tolist()
        if "roberta-large-mnli" == model_name:
            def transform_y(y):
                if y == 0: return "contradiction"
                elif y == 1: return "neutral"
                elif y == 2: return "entailment"
        elif "textattack/xlnet-base-cased-MNLI" == model_name:
            def transform_y(y):
                if y == 0: return "contradiction"
                elif y == 1: return "entailment"
                elif y == 2: return "neutral"
        elif "anirudh21/albert-large-v2-finetuned-mnli" == model_name:
            def transform_y(y):
                if y == 0: return "entailment"
                elif y == 1: return "neutral"
                elif y == 2: return "contradiction"
        elif "microsoft/deberta-large-mnli" == model_name:
            def transform_y(y):
                if y == 0: return "contradiction"
                elif y == 1: return "neutral"
                elif y == 2: return "entailment"
        else:
            raise ValueError(f"mnli model '{model_name}' not recognized/supported")
        return [transform_y(y) for y in answers]

    return mnli_run_model, mnli_encode, mnli_decode

