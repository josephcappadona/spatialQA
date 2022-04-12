import os
from analysis import analyze
from utils import clean_model_name


if __name__ == '__main__':

    from sys import argv
    model_name = argv[1]

    results_dir = 'results'
    results_filename = f"results-{clean_model_name(model_name)}.tsv"
    results_filepath = os.path.join(results_dir, results_filename)
    
    analyze(results_filepath)
    