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
```

`generate.py` outputs `data.tsv`.

`test.py` outputs a results TSV to `results/results-MODEL-NAME.tsv`.

Currently only T5 models are supported.

## TODO

* Update data preprocess for models other than T5
* Separate testing and analysis(?)
* Refactor generation to make it simpler and more scalable