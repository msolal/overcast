import json

import numpy as np

from scipy import stats

from pathlib import Path

from overcast import models
from overcast import datasets
from overcast.models import ensembles
from overcast.visualization import plotting

import seaborn as sns
import matplotlib.pyplot as plt

rc = {
    "figure.constrained_layout.use": True,
    "figure.facecolor": "white",
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.frameon": True,
    "figure.figsize": (6, 6),
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
}
_ = sns.set(style="whitegrid", palette="colorblind", rc=rc)


experiment_dir = Path("/scratch/ms21mmso/output/lr-pacific/jasmin_treatment-None_covariates_outcomes_bins-1/appended-treatment-nn/bohb/tune_step_73927097")
params_path = experiment_dir / "params.json"
checkpoint_dir = experiment_dir = experiment_dir / "checkpoints"