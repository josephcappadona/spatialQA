from script_utils import configure_path
configure_path(__file__)

import os
import csv
import statistics
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
    analysis_filename = f'analysis-{model_name}.tsv'
    analysis_filepath = os.path.join(analysis_dir, analysis_filename)

    summary_dir = 'summary'
    summary_filename = f'summary-{model_name}.tsv'
    summary_filepath = os.path.join(summary_dir, summary_filename)

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(results_filepath, 'r', encoding='utf8') as results_file:

        results_reader = csv.reader(results_file, delimiter='\t', lineterminator='\n')
        headers = next(results_reader)

        with open(analysis_filepath, 'w+t') as analysis_file:

            analysis_headers = ["reasoning_type", "fn_name", "test_id", "num_tests", "num_correct", "test_acc"]
            analysis_writer = csv.writer(analysis_file, delimiter='\t', lineterminator='\n')
            analysis_writer.writerow(analysis_headers)

            with open(summary_filepath, 'w+t') as summary_file:

                summary_headers = ["reasoning_type", "num_tests", "acc_wo_partial_credit", "acc_w_partial_credit", "stdev_acc_w_partial_credit"]
                summary_writer = csv.writer(summary_file, delimiter='\t', lineterminator='\n')
                summary_writer.writerow(summary_headers)

                test_total_counter = defaultdict(lambda: defaultdict(lambda: Counter()))
                test_correct_counter = defaultdict(lambda: defaultdict(lambda: Counter()))
                
                for p, h, e, r_type, fn_name, tid, a, score in results_reader:
                    score = int(score)
                    test_total_counter[r_type][fn_name][tid] += 1
                    test_correct_counter[r_type][fn_name][tid] += score

                # for each reasoning category (motion, distance, etc.)
                for r_type in test_correct_counter:
                    test_accs = []

                    # for each sub-category (motion positive, motion negative, etc.)
                    for fn_name in test_correct_counter[r_type]:

                        # for each test id ()
                        for tid in sorted(list(test_correct_counter[r_type][fn_name].keys())):
                            test_total = test_total_counter[r_type][fn_name][tid]
                            test_correct = test_correct_counter[r_type][fn_name][tid]
                            test_acc = test_correct / test_total
                            test_accs.append(test_acc)
                            analysis_writer.writerow((r_type, fn_name, tid, test_total, test_correct, test_acc))
                    
                    num_tests = len(test_accs)
                    num_perfect = sum([int(uid_acc) for uid_acc in test_accs])
                    perfect_acc = num_perfect / len(test_accs)

                    avg_test_acc = sum(test_accs) / len(test_accs)
                    stdev_test_acc = statistics.pstdev(test_accs)

                    summary_writer.writerow((r_type, num_tests, perfect_acc, avg_test_acc, stdev_test_acc))

                #pprint(subcat_scores)


if __name__ == '__main__':

    from sys import argv

    model_name = argv[1]
    analyze(model_name)
    