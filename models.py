from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
import torch


def get_model(model_name, device):
    if 't5' in model_name:
        return get_t5(model_name, device)
    else:
        return get_auto(model_name, device)


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
    

def get_auto(model_name, device):
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def bart_encode(batch_x):
        return tokenizer.batch_encode_plus(batch_x,
                                            truncation='only_first',
                                            return_tensors="pt",
                                            padding=True)
    
    def bart_run_model(inputs):
        return model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )[0]
    
    def bart_decode(outputs):
        answers = torch.argmax(outputs, axis=1).tolist()
        def transform_y(y):
            if y == 0: return "contradiction"
            elif y == 1: return "neutral"
            elif y == 2: return "entailment"
        return [transform_y(y) for y in answers]

    return bart_run_model, bart_encode, bart_decode

