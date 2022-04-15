from script_utils import configure_path
configure_path(__file__)

import os
import csv
from collections import Counter, defaultdict
from utils import clean_model_name
from pprint import pprint


def analyze(model_name):

    analysis = [model_name] # start analysis log w/ model's name
    cat_scores = []
    subcat_scores = []
    
    model_name = clean_model_name(model_name)

    results_dir = 'results'
    results_filename = f"results-{model_name}.tsv"
    results_filepath = os.path.join(results_dir, results_filename)

    analysis_dir = 'analysis'
    analysis_filename = f'analysis-{model_name}.txt'
    analysis_filepath = os.path.join(analysis_dir, analysis_filename)
    os.makedirs(analysis_dir, exist_ok=True)
    
    with open(results_filepath, 'r', encoding='utf8') as tsv_file:

        with open(analysis_filepath, 'w+t') as analysis_file:

            tsv_reader = csv.reader(tsv_file, delimiter='\t', lineterminator='\n')
            headers = next(tsv_reader)

            total_counter = 0
            correct_counter = 0
            category_total_counter = Counter()
            category_correct_counter = Counter()
            subcategory_total_counter = defaultdict(lambda: Counter())
            subcategory_correct_counter = defaultdict(lambda: Counter())
            uid_total_counter = defaultdict(lambda: defaultdict(lambda: Counter()))
            uid_correct_counter = defaultdict(lambda: defaultdict(lambda: Counter()))

            for p, h, e, r_type, fn_name, id_, a, score in tsv_reader:
                score = int(score)
                total_counter += 1
                correct_counter += score
                category_total_counter[r_type] += 1
                category_correct_counter[r_type] += score
                subcategory_total_counter[r_type][fn_name] += 1
                subcategory_correct_counter[r_type][fn_name] += score
                uid_total_counter[r_type][fn_name][id_] += 1
                uid_correct_counter[r_type][fn_name][id_] += score

            # for each reasoning category (motion, distance, etc.)
            for r_type in category_correct_counter:
                r_score = 0
                cat_correct = category_correct_counter[r_type]
                cat_acc = category_correct_counter[r_type] / category_total_counter[r_type]
                cat_total = category_total_counter[r_type]
                analysis.append(f'{r_type}: {cat_correct} / {cat_total} = {cat_acc:.3g}')

                # for each sub-category (motion positive, motion negative, etc.)
                for fn_name in subcategory_total_counter[r_type]:
                    subcat_acc = subcategory_correct_counter[r_type][fn_name] / subcategory_total_counter[r_type][fn_name]

                    uid_accs = []
                    for id_ in sorted(list(uid_correct_counter[r_type][fn_name].keys())):
                        uid_acc = uid_correct_counter[r_type][fn_name][id_] / uid_total_counter[r_type][fn_name][id_]
                        uid_accs.append(uid_acc)
                    avg_uid_acc = sum(uid_accs) / len(uid_accs)
                    analysis.append(f'\t{fn_name}: avg={avg_uid_acc:.3g}; {subcategory_correct_counter[r_type][fn_name]} / {subcategory_total_counter[r_type][fn_name]} = {subcat_acc:.3g}')
                    sum_perfect = sum([int(uid_acc) for uid_acc in uid_accs])
                    subcat_scores.append((f"{r_type}_{fn_name}", uid_accs, avg_uid_acc))
                        
                    for id_ in sorted(list(uid_correct_counter[r_type][fn_name].keys())):
                        uid_correct = uid_correct_counter[r_type][fn_name][id_]
                        uid_total = uid_total_counter[r_type][fn_name][id_]
                        uid_acc = uid_correct / uid_total
                        analysis.append(f'\t\t{id_}: {uid_correct_counter[r_type][fn_name][id_]} / {uid_total_counter[r_type][fn_name][id_]} = {uid_acc:.3g}')


                        
            analysis.append(f'total: {correct_counter / total_counter}')

            analysis_text = '\n'.join(analysis)
            analysis_file.write(analysis_text)
            print(analysis_text)
            pprint(subcat_scores)


if __name__ == '__main__':

    from sys import argv

    model_name = argv[1]
    analyze(model_name)
    