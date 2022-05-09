from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import openai
from utils import gpt_prompt_template, unifiedqa_MC_template, tfn_decode, transform_gpt3_model_name


def get_model(model_name, device):
    
    if 'unifiedqa' in model_name.lower():
        return get_unifiedqa(model_name, device)

    elif 't5' in model_name.lower():
        return get_t5(model_name, device)

    elif 'mnli' in model_name.lower():
        return get_mnli(model_name, device)

    elif 'snli' in model_name.lower():
        return get_snli(model_name, device)

    elif 'nli' in model_name.lower():
        return get_nli(model_name, device)

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

    def unifiedqa_encode(batch_x):
        transform_x = lambda x: unifiedqa_MC_template.format(premise=x[0], hypothesis=x[1])
        transformed = list(map(transform_x, batch_x))
        return tokenizer(transformed, return_tensors="pt", padding=True)
    
    def unifiedqa_run_model(inputs):
        return model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            do_sample=False
        )
    
    def unifiedqa_decode(outputs):
        batch_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [tfn_decode(y) for y in batch_decoded]

    return unifiedqa_run_model, unifiedqa_encode, unifiedqa_decode


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
    

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_gpt3(model_name):

    api_model_name = transform_gpt3_model_name(model_name)

    def gpt3_encode(batch_x):
        transform_x = lambda x: gpt_prompt_template.format(premise=x[0], hypothesis=x[1])
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
        return [tfn_decode(y) for y in outputs]
    
    return gpt3_run_model, gpt3_encode, gpt3_decode
    

def get_nli(model_name, device):
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def nli_encode(batch_x):
        return tokenizer(batch_x,
                         truncation=True,
                         return_tensors="pt",
                         padding=True)
    
    def nli_run_model(inputs):
        return model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )[0]
    
    def nli_decode(outputs):
        answers = torch.argmax(outputs, axis=1).tolist()
        label_mapping = ['contradiction', 'entailment', 'neutral']
        return [label_mapping[y] for y in answers]

    return nli_run_model, nli_encode, nli_decode
    

def get_snli(model_name, device):
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def nli_encode(batch_x):
        return tokenizer(batch_x,
                         truncation=True,
                         return_tensors="pt",
                         padding=True)
    
    def nli_run_model(inputs):
        return model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )[0]
    
    def nli_decode(outputs):

        answers = torch.argmax(outputs, axis=1).tolist()
        def transform_y(y):
            if y == 0: return "contradiction"
            elif y == 1: return "neutral"
            elif y == 2: return "entailment"
        return [transform_y(y) for y in answers]

    return snli_run_model, snli_encode, snli_decode
    

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
        if model_name in ["textattack/roberta-base-MNLI",
                          "roberta-large-mnli"]:
            def transform_y(y):
                if y == 0: return "contradiction"
                elif y == 1: return "neutral"
                elif y == 2: return "entailment"
        elif "textattack/xlnet-base-cased-MNLI" == model_name:
            def transform_y(y):
                if y == 0: return "contradiction"
                elif y == 1: return "entailment"
                elif y == 2: return "neutral"
        elif model_name in ["anirudh21/albert-large-v2-finetuned-mnli",
                            "Inari/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli",
                            "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
                            "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
                            "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli",
                            "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli",
                            "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli",
                            "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
                            "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"]:
            def transform_y(y):
                if y == 0: return "entailment"
                elif y == 1: return "neutral"
                elif y == 2: return "contradiction"
        elif model_name in ["microsoft/deberta-base-mnli",
                            "microsoft/deberta-large-mnli",
                            "microsoft/deberta-xlarge-mnli",
                            "microsoft/deberta-v2-xxlarge-mnli"]:
            def transform_y(y):
                if y == 0: return "contradiction"
                elif y == 1: return "neutral"
                elif y == 2: return "entailment"
        else:
            raise ValueError(f"mnli model '{model_name}' not recognized/supported")
        return [transform_y(y) for y in answers]

    return mnli_run_model, mnli_encode, mnli_decode

