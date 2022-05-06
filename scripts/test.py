from script_utils import configure_path
configure_path(__file__)

import os
import torch

from dataset import SpatialNLIDataset
from evaluation import evaluate
from models import get_model
from utils import clean_model_name


if __name__ == '__main__':

    from sys import argv
    model_name = argv[1]
    sample = float(argv[2]) if len(argv) > 2 else 1.0

    results_dir = 'results'
    results_filename = f"results-{clean_model_name(model_name)}.tsv"
    results_filepath = os.path.join(results_dir, results_filename)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f'Loading model {model_name}...')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    run_model, encode_batch, decode_batch = get_model(model_name, device)

    print('Model loaded.')

    ds = SpatialNLIDataset('data.tsv', sample=sample)
    evaluate(ds, run_model, encode_batch, decode_batch, results_filepath)