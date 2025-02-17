import json

import numpy as np

from pathlib import Path

from overcast import models
from overcast import datasets
from overcast.models import ensembles

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, experiment_path, name):
        self.experiment_dir = Path(experiment_path)
        self.transformer = True if "daily" in experiment_path else False
        config_path = self.experiment_dir / "config.json"
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.ensemble_dir = self.experiment_dir
        self.name = name

        with open(config_path) as cp:
            config = json.load(cp)

        config["ds_test"]["data_dir"] = "/scratch/ms21mmso/data"
        config["ds_valid"]["data_dir"] = "/scratch/ms21mmso/data"
        config["ds_train"]["data_dir"] = "/scratch/ms21mmso/data"

        self.dataset_name = config.get("dataset_name")
        self.num_components_outcome = config.get("num_components_outcome")
        self.num_components_treatment = config.get("num_components_treatment")
        self.dim_hidden = config.get("dim_hidden")
        self.depth = config.get("depth")
        self.negative_slope = config.get("negative_slope")
        self.beta = config.get("beta")
        self.layer_norm = config.get("layer_norm")
        self.dropout_rate = config.get("dropout_rate")
        self.spectral_norm = config.get("spectral_norm")
        self.learning_rate = config.get("learning_rate")
        self.batch_size = config.get("batch_size")
        self.epochs = config.get("epochs")
        self.ensemble_size = config.get("ensemble_size")
        self.num_heads = config.get("num_heads") if self.transformer is True else None

        self.ds = {
            "test": datasets.DATASETS.get(self.dataset_name)(**config.get("ds_test")),
            "valid": datasets.DATASETS.get(self.dataset_name)(**config.get("ds_valid")),
            "train": datasets.DATASETS.get(self.dataset_name)(**config.get("ds_train")),
        }

        self.target_keys = dict(
            (k, v) for (k, v) in enumerate(self.ds["test"].target_names)
        )

        ensemble = self.load_ensemble()
        treatments = self.load_treatments()
        outcomes = self.load_outcomes()
        apos_ensemble = self.load_apos_ensemble()

        self.ensemble = ensemble
        self.treatments = treatments
        self.outcomes = outcomes
        self.apos_ensemble = apos_ensemble

        means_ensemble = self.get_means_ensemble()
        self.means_ensemble = means_ensemble

        self.apo_limits = {}

    def load_ensemble(self):
        if self.transformer:
            return self.load_transformer_ensemble()
        else:
            return self.load_nn_ensemble()

    def load_transformer_ensemble(self):
        ensemble = []
        for ensemble_id in range(self.ensemble_size):
            model_dir = self.checkpoint_dir / f"model-{ensemble_id}" / "mu"
            model = models.AppendedTreatmentAttentionNetwork(
                job_dir=model_dir,
                dim_input=self.ds["train"].dim_input,
                dim_treatment=self.ds["train"].dim_treatments,
                dim_output=self.ds["train"].dim_targets,
                num_components_outcome=self.num_components_outcome,
                num_components_treatment=self.num_components_treatment,
                dim_hidden=self.dim_hidden,
                depth=self.depth,
                num_heads=self.num_heads,
                negative_slope=self.negative_slope,
                beta=self.beta,
                layer_norm=self.layer_norm,
                spectral_norm=self.spectral_norm,
                dropout_rate=self.dropout_rate,
                num_examples=len(self.ds["train"]),
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                epochs=self.epochs,
                patience=50,
                num_workers=0,
                seed=ensemble_id,
            )
            model.load()
            ensemble.append(model)
        return ensemble

    def load_nn_ensemble(self):
        ensemble = []
        for ensemble_id in range(self.ensemble_size):
            model_dir = self.checkpoint_dir / f"model-{ensemble_id}" / "mu"
            model = models.AppendedTreatmentNeuralNetwork(
                job_dir=model_dir,
                architecture="resnet",
                dim_input=self.ds["train"].dim_input,
                dim_treatment=self.ds["train"].dim_treatments,
                dim_output=self.ds["train"].dim_targets,
                num_components_outcome=self.num_components_outcome,
                num_components_treatment=self.num_components_treatment,
                dim_hidden=self.dim_hidden,
                depth=self.depth,
                negative_slope=self.negative_slope,
                beta=self.beta,
                layer_norm=self.layer_norm,
                spectral_norm=self.spectral_norm,
                dropout_rate=self.dropout_rate,
                num_examples=len(self.ds["train"]),
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                epochs=self.epochs,
                patience=self.epochs,
                num_workers=0,
                seed=ensemble_id,
            )
            model.load()
            ensemble.append(model)
        return ensemble

    def load_treatments(self):
        if self.transformer:
            treatments = np.concatenate(self.ds["train"].treatments, axis=0)
            treatments = self.ds["train"].treatments_xfm.inverse_transform(treatments)
            treatments = np.quantile(treatments, q=np.arange(0, 1 + 1 / 32, 1 / 32),)[:-1]
        else:
            treatments = np.quantile(
                self.ds["train"].treatments_xfm.inverse_transform(
                    self.ds["train"].treatments
                ),
                q=np.arange(0, 1 + 1 / 32, 1 / 32),
            )[:-1]        
        return treatments

    def load_outcomes(self):
        if self.transformer:
            df_test = self.ds["test"].data_frame
            observed_outcomes = df_test.to_numpy()[:, -4:]
        else: 
            observed_outcomes = self.ds["test"].targets_xfm.inverse_transform(self.ds["test"].targets)
        return observed_outcomes
    
    def load_apos_ensemble(self):
        apos_ensemble_path = self.ensemble_dir / "apos_ensemble.npy"
        if not apos_ensemble_path.exists():
            capos_ensemble = ensembles.predict_capos(
                ensemble=self.ensemble,
                dataset=self.ds["test"],
                treatments=self.treatments,
                batch_size=1 if self.transformer else 20000,
            )
            apos_ensemble = capos_ensemble.mean(2)
            np.save(apos_ensemble_path, apos_ensemble)
        else:
            apos_ensemble = np.load(apos_ensemble_path)
        return apos_ensemble

    def get_apo_limits(self, log_lambda, from_scratch=False):
        apo_limits_path = self.ensemble_dir / f"apo_limits_{log_lambda}.npy"
        if not apo_limits_path.exists() or from_scratch:
            lower_capos, upper_capos = ensembles.predict_intervals(
                ensemble=self.ensemble,
                dataset=self.ds["test"],
                treatments=self.treatments,
                log_lambda=log_lambda,
                num_samples=100,
                batch_size=1 if self.transformer else 10000,
            )
            lower_apos = np.expand_dims(lower_capos.mean(2), 0)
            upper_apos = np.expand_dims(upper_capos.mean(2), 0)
            apo_limits = np.concatenate([lower_apos, upper_apos], axis=0)
            np.save(apo_limits_path, apo_limits)
        else:
            apo_limits = np.load(apo_limits_path)
        self.apo_limits[log_lambda] = apo_limits

    def get_means_ensemble(self): 
        return ensembles.predict_mean(self.ensemble, self.ds["test"], batch_size=None)

tr_pacific_without_w500 = Experiment(
    "/scratch/ms21mmso/output/nice/jasmin-daily-four_outputs_liqcf_pacific_treatment-AOD_covariates-RH900-RH850-RH700-LTS-EIS-SST_outcomes-re_bins-1/appended-treatment-transformer/dh-256_nco-24_nct-24_dp-3_nh-4_ns-0.01_bt-0.0_ln-False_dr-0.5_sn-0.0_lr-0.0002_bs-32_ep-500",
    r"Pacific without $\omega500$",
)

tr_pacific = Experiment(
    "/scratch/ms21mmso/output/nice/jasmin-daily-four_outputs_liqcf_pacific_treatment-AOD_covariates-RH900-RH850-RH700-LTS-EIS-W500-SST_outcomes-re-COD-CWP-LPC_bins-1/appended-treatment-transformer/dh-128_nco-22_nct-27_dp-3_nh-8_ns-0.28_bt-0.0_ln-False_dr-0.42_sn-0.0_lr-0.0001_bs-128_ep-500",
    "LR Pacific with context",
)

tr_pacific_without_rh = Experiment(
    "/scratch/ms21mmso/output/nice/jasmin-daily-four_outputs_liqcf_pacific_treatment-AOD_covariates-LTS-EIS-W500-SST_outcomes-re-COD-CWP-LPC_bins-1/appended-treatment-transformer/dh-128_nco-22_nct-27_dp-3_nh-8_ns-0.28_bt-0.0_ln-False_dr-0.42_sn-0.0_lr-0.0001_bs-128_ep-500", 
    r"Pacific without RH",
)

_ = tr_pacific.get_apo_limits(0.25, from_scratch=True)
_ = tr_pacific.get_apo_limits(0.15, from_scratch=True)
_ = tr_pacific_without_rh.get_apo_limits(0.02, from_scratch=True)