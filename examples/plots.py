import json

import numpy as np

from scipy import stats

from pathlib import Path

from overcast import models
from overcast import datasets
from overcast.models import ensembles
from overcast.visualization import plotting

from sklearn.preprocessing import MinMaxScaler

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

TARGET_KEYS = {
    "Nd": "N_d", 
    "re": r"$r_e$", 
    "COD": r"$\tau$",
    "CWP": "CWP", 
    "LPC": r"$CF_w$",
}

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


def plot_scatterplot(experiment, idx_outcome, savepath=None):
    plt.plot(figsize=(6, 6))
    qs = np.quantile(experiment.outcomes[:, idx_outcome], [0.01, 0.99])
    domain = np.arange(qs[0], qs[1], 0.01)
    means_ensemble = experiment.get_means_ensemble()
    slope, intercept, r, p, stderr = stats.linregress(
        experiment.outcomes[:, idx_outcome], means_ensemble.mean(0)[:, idx_outcome]
    )
    sns.scatterplot(x=experiment.outcomes[:, idx_outcome], y=means_ensemble.mean(0)[:, idx_outcome], s=0.5)
    plt.plot(domain, domain, c="C1")
    plt.plot(domain, domain * slope + intercept, c="C2", label=f"$r^2$={r**2:.03f}")
    plt.xlim(qs)
    plt.ylim(qs)
    plt.xlabel(f"{TARGET_KEYS[experiment.target_keys[idx_outcome]]} true")
    plt.ylabel(f"{TARGET_KEYS[experiment.target_keys[idx_outcome]]} predicted")
    plt.legend(loc="upper left")
    if savepath is not None:
        plt.savefig(savepath)


def plot_apos(reference_experiment, comparison_experiments, apo_limits_log_lambda_list):
    alpha = 0.05
    _, ax = plt.subplots(2, 2, figsize=(12, 12))
    for idx_outcome in range(len(reference_experiment.target_keys)):
        i, j = idx_outcome // 2, idx_outcome % 2
        # plot comparison experiments APOs
        for k in range(len(comparison_experiments)):
            scaler = MinMaxScaler()
            scaler = scaler.fit(
                comparison_experiments[k]
                .apos_ensemble[idx_outcome]
                .mean(0)
                .reshape(-1, 1)
            )
            _ = ax[i][j].plot(
                comparison_experiments[k].treatments,
                scaler.transform(
                    comparison_experiments[k]
                    .apos_ensemble[idx_outcome]
                    .mean(0)
                    .reshape(-1, 1)
                ),
                label=comparison_experiments[k].name,
            )
        reference_scaler = MinMaxScaler()
        reference_scaler = reference_scaler.fit(
            reference_experiment.apos_ensemble[idx_outcome].mean(0).reshape(-1, 1)
        )
        _ = ax[i][j].plot(
            reference_experiment.treatments,
            scaler.transform(
                reference_experiment.apos_ensemble[idx_outcome].mean(0).reshape(-1, 1)
            ),
            label=reference_experiment.name,
        )
        for l in apo_limits_log_lambda_list:
            if l not in reference_experiment.apo_limits:
                reference_experiment.get_apo_limits(l)
        for idx in range(len(apo_limits_log_lambda_list)-1):
            l = apo_limits_log_lambda_list[idx]
            next_l = apo_limits_log_lambda_list[idx + 1]
            _ = ax[i][j].fill_between(
                x=reference_experiment.treatments,
                y1=reference_scaler.transform(
                    np.quantile(
                        reference_experiment.apo_limits[l][1][idx_outcome],
                        1 - alpha / 2,
                        axis=0,
                    ).reshape(-1, 1)
                ).flatten(),
                y2=reference_scaler.transform(
                    np.quantile(
                        reference_experiment.apo_limits[next_l][1][idx_outcome],
                        alpha / 2,
                        axis=0,
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                label=r"$\Lambda=$" + f"{np.exp(l):.01f}",
            )
            _ = ax[i][j].fill_between(
                x=reference_experiment.treatments,
                y1=reference_scaler.transform(
                    np.quantile(
                        reference_experiment.apo_limits[l][0][idx_outcome],
                        1 - alpha / 2,
                        axis=0,
                    ).reshape(-1, 1)
                ).flatten(),
                y2=reference_scaler.transform(
                    np.quantile(
                        reference_experiment.apo_limits[next_l][0][idx_outcome],
                        alpha / 2,
                        axis=0,
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                label=r"$\Lambda=$" + f"{np.exp(l):.01f}",
            )
        last = apo_limits_log_lambda_list[-1]
        _ = ax[i][j].fill_between(
            x=reference_experiment.treatments,
            y1=reference_scaler.transform(
                np.quantile(
                    reference_experiment.apo_limits[last][1][idx_outcome],
                    1 - alpha / 2,
                    axis=0,
                ).reshape(-1, 1)
            ).flatten(),
            y2=reference_scaler.transform(
                np.quantile(
                    reference_experiment.apos_ensemble[idx_outcome], alpha / 2, axis=0
                ).reshape(-1, 1)
            ).flatten(),
            alpha=0.2,
            label=r"$\Lambda=$" + f"{np.exp(l):.01f}",
        )
        _ = ax[i][j].fill_between(
            x=reference_experiment.treatments,
            y1=reference_scaler.transform(
                np.quantile(
                    reference_experiment.apo_limits[last][0][idx_outcome],
                    1 - alpha / 2,
                    axis=0,
                ).reshape(-1, 1)
            ).flatten(),
            y2=reference_scaler.transform(
                np.quantile(
                    reference_experiment.apos_ensemble[idx_outcome], alpha / 2, axis=0
                ).reshape(-1, 1)
            ).flatten(),
            alpha=0.2,
            label=r"$\Lambda=$" + f"{np.exp(last):.01f}",
        )
        _ = ax[i][j].fill_between(
            x=reference_experiment.treatments,
            y1=np.quantile(reference_experiment.apos_ensemble[idx_outcome], 1 - alpha / 2, axis=0),
            y2=np.quantile(reference_experiment.apos_ensemble[idx_outcome], alpha / 2, axis=0),
            alpha=0.2,
            label=r"$\Lambda \to 1.0 $",
        )
        _ = ax[i][j].legend(title=r"$\alpha=$" + f"{alpha}", loc="upper left",)
        _ = ax[i][j].set_xlabel(reference_experiment.ds["train"].treatment_names[0])
        _ = ax[i][j].set_ylabel(reference_experiment.target_keys[idx_outcome])
    plt.savefig(f"results/{reference_experiment.name}-{apo_limits_log_lambda_list[0]}.pdf")

tr_pacific = Experiment(
    "/scratch/ms21mmso/output/nice/jasmin-daily-four_outputs_liqcf_pacific_treatment-AOD_covariates-RH900-RH850-RH700-LTS-EIS-W500-SST_outcomes-re-COD-CWP-LPC_bins-1/appended-treatment-transformer/dh-128_nco-22_nct-27_dp-3_nh-8_ns-0.28_bt-0.0_ln-False_dr-0.42_sn-0.0_lr-0.0001_bs-128_ep-500",
    "Pacific",
)

plot_scatterplot(tr_pacific, 0, '/scratch/ms21mmso/results/scatterplot_tr_lrp_re.pdf')