# spatialQA

## Setup

```bash
python3 -m pip install -r requirements.txt
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
# Combines result of different model
python scripts/make_dataframes.py analysis/ summary/

# Generate figures
python scripts/make_figures.py analysis/df_analysis.csv summary/df_summary.csv 
```

`make_dataframes.py` outputs a dataframe of combined analysis results to `analysis/df_analysis.csv` and a dataframe of combined summary results to `summary/df_summary.csv` 

`make_figures.py` output the figures to `figures`

## TODO

* Refactor generation to make it simpler and more scalable
