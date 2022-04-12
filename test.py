import os
from data import SpatialNLIDataset
import torch

from evaluation import evaluate
from analysis import analyze
from models import get_model
from utils import clean_model_name


if __name__ == '__main__':

    from sys import argv
    model_name = argv[1]
    sample = 1.0

    results_dir = 'results'
    results_filename = f"results-{clean_model_name(model_name)}.tsv"
    results_filepath = os.path.join(results_dir, results_filename)
    os.makedirs(results_dir, exist_ok=True)
    
    print('Loading model...')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    run_model, encode_batch, decode_batch = get_model(model_name, device)

    print('Model loaded.')

    ds = SpatialNLIDataset('data.tsv', sample=sample)
    evaluate(ds, run_model, encode_batch, decode_batch, results_filepath)