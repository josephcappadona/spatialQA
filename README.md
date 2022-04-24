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


## Computing graphs

```bash
# generate plot of analysis
python graph/analysis.py

# generate plot of summary
python graph/summary.py
```

`analysis.py` outputs a dataframe of combined analysis result to `analysis/df.analysis.csv` and a folder that contains graph created by the dataframe `df.analysis` to `analysis/output_fig_analysis`.

`summary.py` outputs a dataframe of combined summary result to `summary/df.summary.csv` and a folder that contains graph created by the dataframe `df.summary` to `summary/output_fig_summary`.

## TODO

* Refactor generation to make it simpler and more scalable
