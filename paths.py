"""Central location of every input and output path used by the notebooks.

UK Biobank data are confidential. Participant-level files (raw extracts, the preprocessed
phenotype table, split membership, MLTE embeddings, checkpoints and out-of-fold predictions)
are kept OUTSIDE this repository by default. Only aggregate result tables are written into the
project folder (``results/``, which is git-ignored as well).

Every path can be overridden with an environment variable, e.g.

    export UKB_DATA_ROOT=/mnt/ukb/raw
    export UKB_PREPROCESSED_DIR=/mnt/ukb/preprocessed
"""
import os
import warnings

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw UK Biobank extracts (read only): phenotypes/*.txt.gz, ukb_showcase.xlsx, *.codings.csv, *.data_dictionary.csv
DATA_ROOT = os.environ.get("UKB_DATA_ROOT", "/data/UKB")

# Participant-level intermediate files: phenotypes.tsv, splits_all_repeats.parquet, patient_embeddings.parquet,
# ckpt_*/ checkpoint folders and preds_*.pkl (out-of-fold predictions)
PREPROCESSED_DIR = os.environ.get("UKB_PREPROCESSED_DIR", "/preprocessed/UKB")

# Aggregate results only (per-fold metrics, bootstrap and repeated-CV tables); read by 03_manuscript_statistics.ipynb
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(PROJECT_DIR, "results"))


def _inside_project(path):
    path = os.path.realpath(path)
    return os.path.commonpath([path, os.path.realpath(PROJECT_DIR)]) == os.path.realpath(PROJECT_DIR)


if _inside_project(PREPROCESSED_DIR) or _inside_project(DATA_ROOT):
    warnings.warn(
        "UKB_DATA_ROOT or UKB_PREPROCESSED_DIR points inside the project folder. These folders hold "
        "participant-level data: make sure they are git-ignored and never published.",
        stacklevel=2,
    )
