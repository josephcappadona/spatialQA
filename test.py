import csv
import torch
import os
from data import SpatialNLIDataset
from sklearn.metrics import accuracy_score as accuracy
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm



def evaluate(ds, model, tokenizer, results_fn="results.tsv"):
    # adapted from https://huggingface.co/docs/transformers/model_doc/t5#training
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    tsv_file = open(results_fn, 'w+', encoding='utf8', newline='')
    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
    tsv_writer.writerow(['premise', 'hypothesis', 'entailment', 'reasoning_type', 'function_name', 'answer', 'correct'])

    count = 0
    total = 0

    for i, (batch_x, batch_y, batch_m) in enumerate(ds.batch(
        batch_size=32,
    )):
        if i % 10 == 0: print(i)
        transform = lambda x: f"mnli hypothesis: {x[1]} premise: {x[0]}" # task setup for T5
        batch_x_transformed = list(map(transform, batch_x))
        
        inputs = tokenizer(batch_x_transformed, return_tensors="pt", padding=True)
        output_sequences = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            do_sample=False,
        )
        answers = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        acc = accuracy(answers, batch_y)
        count += acc * len(answers)
        total += len(answers)

        for (p, h), e, (r_type, fn_name), a in zip(batch_x, batch_y, batch_m, answers):
            tsv_writer.writerow([p, h, e, r_type, fn_name, a, int(e == a)])
        
    print('Acc:', count / total)


def analyze(results_fn):
    with open(results_fn, 'r', encoding='utf8') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t', lineterminator='\n')
        headers = next(tsv_reader)

        from collections import Counter, defaultdict
        category_total_counter = Counter()
        category_correct_counter = Counter()
        category_subcategory_total_counter = defaultdict(lambda: Counter())
        category_subcategory_correct_counter = defaultdict(lambda: Counter())

        for p, h, e, r_type, fn_name, a, score in tsv_reader:
            score = int(score)
            category_total_counter[r_type] += 1
            category_correct_counter[r_type] += score
            category_subcategory_total_counter[r_type][fn_name] += 1
            category_subcategory_correct_counter[r_type][fn_name] += score
        from pprint import pprint
        #pprint(category_total_counter)
        #pprint(category_correct_counter)
        #pprint(category_subcategory_total_counter)
        #pprint(category_subcategory_correct_counter)

        for r_type in category_correct_counter:
            print(f'{r_type}: {category_correct_counter[r_type]} / {category_total_counter[r_type]} = {category_correct_counter[r_type] / category_total_counter[r_type]:.3g}')
            for fn_name in category_subcategory_total_counter[r_type]:
                print(f'\t{fn_name}: {category_subcategory_correct_counter[r_type][fn_name]} / {category_subcategory_total_counter[r_type][fn_name]} = {category_subcategory_correct_counter[r_type][fn_name] / category_subcategory_total_counter[r_type][fn_name]:.3g}')

if __name__ == '__main__':

    from sys import argv
    model_name = argv[1]
    sample = 1.0

    results_dir = 'results'
    results_filename = f"results-{model_name.replace('/', '-')}.tsv"
    results_filepath = os.path.join(results_dir, results_filename)
    os.makedirs(results_dir, exist_ok=True)
    
    print('Loading model...')
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print('Model loaded.')

    ds = SpatialNLIDataset('data.tsv', sample=sample)
    evaluate(ds, model, tokenizer, results_filepath)
    analyze(results_filepath)
    