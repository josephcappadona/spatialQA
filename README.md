# spatialQA

## Setup

```bash
python3 -m pip install transformers sentencepiece torch
```

## Usage

```bash
# generate data
python generate.py

# test model
python test.py t5-small

# analyze results
python analyze.py t5-small
```

`generate.py` outputs `data.tsv`.

`test.py` outputs a results TSV to `results/results-MODEL-NAME.tsv`.

`analyze.py` prints an analysis of model performance broken down by question category (motion, distance, etc.) and sub-category (motion_positive, motion_negative, etc.).

Currently only T5, BART, and RoBERTa are supported.

## TODO

* Add more models (GPT-3, UnifiedQAv2)
* Refactor generation to make it simpler and more scalable
* Add more examples
* Add support for multiple correct answers (e.g., neutral or entailment)