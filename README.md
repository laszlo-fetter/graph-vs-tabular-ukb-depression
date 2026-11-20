# Do Graph Representations Add Value Beyond Tabular Disease-History Features for Depression Prediction?

_A controlled comparison of patient-similarity graphs and direct disease-history features for predicting a first recorded depressive episode (ICD-10 F32) in UK Biobank._

## Overview

Disease histories can be given to a model as tabular features, or used to build patient-similarity graphs over which baseline features are propagated. This repository contains the code for a controlled, within-cohort comparison of the two approaches in 68,618 UK Biobank participants. The comparison includes four disease-history graphs, control graphs, onset-age and disease-count features, and tuned and default classifiers.

The main finding is that disease-history graphs carry predictive information that random, age-and-sex and shuffled-history control graphs do not. However, direct onset-age and disease-count features perform at least as well as the graphs, and adding graph-propagated features to onset ages gives no gain of practical size.

Full text: _manuscript under review — link will be added on publication._

## Method in brief

- **Cohort:** White British UK Biobank participants passing genetic QC, baseline on or before 31 March 2008, without F32 at baseline. Outcome: a first recorded F32 within 15 years.
- **Baseline features:** 62 covariates (demographics, deprivation, anthropometry, lifestyle, social, cognitive, NMR omega fatty acids, parental illness, genotyping array, 20 genetic PCs).
- **Disease history:** first-record age of 207 three-character ICD-10 categories, masked at baseline.
- **Similarity graphs** (k = 50, exact cosine search, row-normalised, inductive label-free propagation `[X, AX, A¹⁰X]`):
  - **OAO**: onset-age overlap (weighted Jaccard × onset-timing agreement)
  - **MDBT**: multiresolution disease-burden trajectory (chapter-level cumulative burden and rates)
  - **DCEG**: disease co-occurrence embedding graph (SVD of the training co-occurrence matrix, recency-weighted)
  - **MLTE**: learned patient embedding from a masked language model (trained externally, per fold)
- **Controls:** random, age-and-sex, shuffled-history (DCEG), node-shuffled, feature-kNN.
- **Classifiers:** nested-CV tuned L1/L2 logistic regression (primary), default L2 logistic regression, default and nested-tuned CatBoost; logistic recalibration on a validation partition.
- **Evaluation:** 10-fold CV (plus 2 extra repeats as a sensitivity analysis), AUROC/AUPR/calibration, corrected resampled-*t* intervals and a paired patient-level bootstrap.

All disease weights, embeddings, graphs, imputation, scaling and model fits are estimated on training partitions only.

## Repository layout

| File | Purpose |
|---|---|
| `paths.py` | All input/output locations (overridable with environment variables) |
| `01_preprocessing.ipynb` | Builds the participant-level phenotype table from the raw UK Biobank extracts |
| `02_experiments.ipynb` | Cohort selection, splits, graph construction, model fitting, bootstrap, repeated CV |
| `03_manuscript_statistics.ipynb` | Recomputes every number in the manuscript from the aggregate result files and draws the forest plot |
| `preprocessing_helpers.py` | Variable recoding helpers |
| `experiments_helpers.py` | Calibration metrics and logistic recalibration |

## Data access and confidentiality

UK Biobank data are **not** included and cannot be shared. They are available to approved researchers through the [UK Biobank access process](https://www.ukbiobank.ac.uk/enable-your-research). This analysis used Application 1602.

The code keeps participant-level and aggregate files apart:

| Location | Default | Contents | Commit? |
|---|---|---|---|
| `UKB_DATA_ROOT` | `/data/UKB` | raw UK Biobank extracts (read only) | **never** |
| `UKB_PREPROCESSED_DIR` | `/preprocessed/UKB` | `phenotypes.tsv`, `splits_all_repeats.parquet`, `patient_embeddings.parquet`, `ckpt_*/`, `preds_*.pkl` | **never** |
| `RESULTS_DIR` | `./results` | per-fold metrics, bootstrap and repeated-CV tables (aggregate only) | git-ignored by default |

## Expected input files

Under `UKB_DATA_ROOT`:

```
phenotypes/
  Age.txt.gz  CF.txt.gz  Other.txt.gz  Adversities.txt.gz  PA.txt.gz
  NMR_1.txt.gz ... NMR_5.txt.gz
  Diseases_1.txt.gz ... Diseases_4.txt.gz  Other_Diseases.txt.gz
ukb_showcase.xlsx           # field lists per extract (sheets: CF, Adversities, Diseases_1..4, Other_Diseases)
*.codings.csv               # UK Biobank RAP dataset codings
*.data_dictionary.csv       # UK Biobank RAP data dictionary
```

The phenotype files are tab-separated, keyed by `eid`, with UK Biobank RAP column names (`p<field>_i<instance>[_a<array>]`). `ukb_showcase.xlsx` lists the fields of each extract: `Field ID`, plus `Direction` (CF, Adversities), `Group` (Adversities), `Description` = ICD-10 code (Diseases_*) and `Name` (Other_Diseases).

### MLTE embeddings (external)

The MLTE embeddings come from a separately trained masked language model, which is not part of this repository. One model is trained per (repeat, fold) on that fold's training participants only. `02_experiments.ipynb` exports the splits it expects to `UKB_PREPROCESSED_DIR/splits_all_repeats.parquet` and reads the embeddings back from `UKB_PREPROCESSED_DIR/patient_embeddings.parquet`, with columns:

| Column | Meaning |
|---|---|
| `IID` | participant ID |
| `repeat`, `fold` | which split the model was trained on |
| `role` | `train` / `val` / `test` |
| `train_hash` | copied from the split file; MD5 of the sorted, comma-joined training IIDs |
| `assessment_date` | baseline date (consistency check) |
| `embedding` | 256-d vector |

The notebook stops if `train_hash`, the training IIDs or the assessment dates do not match its own splits. To run without MLTE, remove `mlte_50`, `mlte_50_onset` and `raw_mltevec` from `ALL_ARMS` and `RCV_MAIN_ARMS` in `02_experiments.ipynb`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optionally, point the code at other locations:

```bash
export UKB_DATA_ROOT=/path/to/ukb/raw
export UKB_PREPROCESSED_DIR=/path/outside/repo/preprocessed
```

## Reproducing the experiments

Run the notebooks in order, from the repository root:

1. **`01_preprocessing.ipynb`** writes `phenotypes.tsv`. The cells after the export are exploratory and are not used in the manuscript.
2. **`02_experiments.ipynb`**
   1. Run the cells up to and including the split export. This writes `splits_all_repeats.parquet`.
   2. Train the MLTE models on those splits (externally) and place `patient_embeddings.parquet` in `UKB_PREPROCESSED_DIR`.
   3. Run the remaining cells: the main run (all arms × 4 classifiers × 10 folds), the patient-level bootstrap, and the repeated cross-validation.

   The main run and the repeated-CV cell checkpoint each fold to `UKB_PREPROCESSED_DIR/ckpt_*` and resume automatically (`RESUME = True`). Checkpoints are reused only if `BUILD_TAG` and the arm/classifier plan match exactly. The repeated-CV cell stops cleanly after `RCV_MAX_MINUTES` and can be re-run to continue. Peak memory is about one representation's feature matrix plus a few cached blocks (around 1–2 GB).
3. **`03_manuscript_statistics.ipynb`** reads the CSV files in `results/` and recomputes every table and quoted number. The claims register at the end compares each value with the manuscript. Set `WRITE_FILES = True` to export all tables (Excel) and the forest plot.

All seeds are fixed (`SEED = 12345`; exact neighbour search with seeded tie-breaking).

> **Earlier runs:** in previous versions, `preds_*.pkl` and the result CSVs were written to the working directory. To reuse an existing run, move `preds_*.pkl` into `UKB_PREPROCESSED_DIR` and the CSV files into `results/`. Checkpoints already in `/preprocessed/UKB` are picked up unchanged.

## Citation

If you use this code, please cite:

```bibtex
@unpublished{fetter2026graphtabular,
  author = {Fetter, László and Gézsi, András},
  title  = {Do Graph Representations Add Value Beyond Tabular Disease-History Features for Depression Prediction? A Controlled Comparison in UK Biobank},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## Acknowledgements

This research was conducted using the UK Biobank Resource under Application Number 1602. The authors thank Mátyás Antal for providing the MLTE embeddings. The project, supported by the Doctoral Excellence Fellowship Programme (DCEP), is funded by the National Research Development and Innovation Fund of the Ministry of Science and Technology and the Budapest University of Technology and Economics.

## Contact

If you have any questions, please contact László Fetter at fetter (*) mit * bme * hu.
