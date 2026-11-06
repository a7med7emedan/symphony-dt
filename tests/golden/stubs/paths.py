"""Stub of the study's `paths` module: constants and synthetic file locations only."""
import os
from pathlib import Path

REPO = Path(os.environ["SDT_GOLDEN_ROOT"])
RESULTS = REPO / "results"
PREPROC_DIR = RESULTS / "02_preprocessing"
SPLITS_DIR = RESULTS / "01_splits"
DAYS_PER_YEAR = 365.25
UPDRS3_SUPPORT = (0, 132)
MOCA_SUPPORT = (0, 30)
SCOPA_SUPPORT = (0, 69)
SPLIT_KEY_BAN = frozenset({"genotype_assay_available"})
MASTER_SEED = 20260729
SEX_LABELS = {0: "Female", 1: "Male"}
