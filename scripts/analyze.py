from script_utils import configure_path
configure_path(__file__)

import os
import csv
from collections import Counter, defaultdict
from utils import clean_model_name


def analyze(model_name):

    analysis = [model_name] # start analysis log w/ model's name
    
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
            category_subcategory_total_counter = defaultdict(lambda: Counter())
            category_subcategory_correct_counter = defaultdict(lambda: Counter())

            for p, h, e, r_type, fn_name, a, score in tsv_reader:
                score = int(score)
                total_counter += 1
                correct_counter += score
                category_total_counter[r_type] += 1
                category_correct_counter[r_type] += score
                category_subcategory_total_counter[r_type][fn_name] += 1
                category_subcategory_correct_counter[r_type][fn_name] += score

            # for each reasoning category (motion, distance, etc.)
            for r_type in category_correct_counter:
                analysis.append(f'{r_type}: {category_correct_counter[r_type]} / {category_total_counter[r_type]} = {category_correct_counter[r_type] / category_total_counter[r_type]:.3g}')

                # for each sub-category (motion positive, motion negative, etc.)
                for fn_name in category_subcategory_total_counter[r_type]:
                    analysis.append(f'\t{fn_name}: {category_subcategory_correct_counter[r_type][fn_name]} / {category_subcategory_total_counter[r_type][fn_name]} = {category_subcategory_correct_counter[r_type][fn_name] / category_subcategory_total_counter[r_type][fn_name]:.3g}')
            analysis.append(f'total: {correct_counter / total_counter}')

            analysis_text = '\n'.join(analysis)
            analysis_file.write(analysis_text)
            print(analysis_text)


if __name__ == '__main__':

    from sys import argv

    model_name = argv[1]
    analyze(model_name)
    