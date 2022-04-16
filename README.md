# spatialQA

## Setup

```bash
python3 -m pip install transformers sentencepiece torch
```

## Usage

```bash
# generate data
python scripts/generate.py

# test model
python scripts/test.py t5-small

# analyze results
python scripts/analyze.py t5-small
```

`generate.py` outputs `data.tsv`.

`test.py` outputs a results TSV to `results/results-MODEL-NAME.tsv`.

`analyze.py` outputs a summary TSV to `summary/summary-MODEL-NAME.tsv` and an analysis TSV to `anlysis/analysis-MODEL-NAME.tsv`.


## TODO

* Refactor generation to make it simpler and more scalable
