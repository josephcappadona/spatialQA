import csv
from collections import Counter, defaultdict

def analyze(results_fn):
    with open(results_fn, 'r', encoding='utf8') as tsv_file:
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

        for r_type in category_correct_counter:
            print(f'{r_type}: {category_correct_counter[r_type]} / {category_total_counter[r_type]} = {category_correct_counter[r_type] / category_total_counter[r_type]:.3g}')
            for fn_name in category_subcategory_total_counter[r_type]:
                print(f'\t{fn_name}: {category_subcategory_correct_counter[r_type][fn_name]} / {category_subcategory_total_counter[r_type][fn_name]} = {category_subcategory_correct_counter[r_type][fn_name] / category_subcategory_total_counter[r_type][fn_name]:.3g}')
        print(f'\ntotal: {correct_counter / total_counter}')
