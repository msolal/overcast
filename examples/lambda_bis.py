import json

import numpy as np

from scipy import stats
from sklearn.preprocessing import minmax_scale, MinMaxScaler

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

experiment_dict = {
    "pacific": Path(
        "/users/ms21mmso/msc-project/overcast/output/jasmin-four_outputs_liqcf_atlantic_treatment-AOD_covariates-RH900-RH850-RH700-LTS-EIS-W500-SST_outcomes-re-COD-CWP-LPC_bins-1/appended-treatment-nn/dh-96_nco-3_nct-9_dp-5_ns-0.11_bt-0.0_ln-False_dr-0.04_sn-0.0_lr-0.0002_bs-224_ep-9"
    ),
    "atlantic": Path(
        "/users/ms21mmso/msc-project/overcast/output/jasmin-four_outputs_liqcf_atlantic_treatment-AOD_covariates-RH900-RH850-RH700-LTS-EIS-W500-SST_outcomes-re-COD-CWP-LPC_bins-1/appended-treatment-nn/dh-96_nco-3_nct-9_dp-5_ns-0.11_bt-0.0_ln-False_dr-0.04_sn-0.0_lr-0.0002_bs-224_ep-9"
    ),
}

transformer = False
comparison_key = "pacific"
comparison_name = "jasmin-pacific-four_outputs_liqcf"

ds_dict = {}
treatments_dict = {}
ensemble_dict = {}
apo_ensemble_dict = {}
apo_limits_infty_dict = {}
apo_limits_1_dict = {}
apo_limits_2_dict = {}
apo_limits_3_dict = {}
apo_limits_4_dict = {}
target_keys_dict = {}

for k, v in experiment_dict.items():
    experiment_dir = v
    config_path = experiment_dir / "config.json"
    checkpoint_dir = experiment_dir / "checkpoints"
    ensemble_dir = experiment_dir

    with open(config_path) as cp:
        config = json.load(cp)

    dataset_name = config.get("dataset_name")
    num_components_outcome = config.get("num_components_outcome")
    num_components_treatment = config.get("num_components_treatment")
    dim_hidden = config.get("dim_hidden")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    beta = config.get("beta")
    layer_norm = config.get("layer_norm")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    ensemble_size = config.get("ensemble_size")
    num_heads = config.get("num_heads") if transformer is True else None

    ds = {
        "test": datasets.DATASETS.get(dataset_name)(**config.get("ds_test")),
        "valid": datasets.DATASETS.get(dataset_name)(**config.get("ds_valid")),
        "train": datasets.DATASETS.get(dataset_name)(**config.get("ds_train")),
    }

    target_keys = dict((k, v) for (k, v) in enumerate(ds["test"].target_names))

    if transformer:
        ensemble = []
        for ensemble_id in range(ensemble_size):
            model_dir = checkpoint_dir / f"model-{ensemble_id}" / "mu"
            model = models.AppendedTreatmentAttentionNetwork(
                job_dir=model_dir,
                dim_input=ds["train"].dim_input,
                dim_treatment=ds["train"].dim_treatments,
                dim_output=ds["train"].dim_targets,
                num_components_outcome=num_components_outcome,
                num_components_treatment=num_components_treatment,
                dim_hidden=dim_hidden,
                depth=depth,
                num_heads=num_heads,
                negative_slope=negative_slope,
                beta=beta,
                layer_norm=layer_norm,
                spectral_norm=spectral_norm,
                dropout_rate=dropout_rate,
                num_examples=len(ds["train"]),
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                patience=50,
                num_workers=0,
                seed=ensemble_id,
            )
            model.load()
            ensemble.append(model)
    else:
        ensemble = []
        for ensemble_id in range(ensemble_size):
            model_dir = checkpoint_dir / f"model-{ensemble_id}" / "mu"
            model = models.AppendedTreatmentNeuralNetwork(
                job_dir=model_dir,
                architecture="resnet",
                dim_input=ds["train"].dim_input,
                dim_treatment=ds["train"].dim_treatments,
                dim_output=ds["train"].dim_targets,
                num_components_outcome=num_components_outcome,
                num_components_treatment=num_components_treatment,
                dim_hidden=dim_hidden,
                depth=depth,
                negative_slope=negative_slope,
                beta=beta,
                layer_norm=layer_norm,
                spectral_norm=spectral_norm,
                dropout_rate=dropout_rate,
                num_examples=len(ds["train"]),
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                patience=epochs,
                num_workers=0,
                seed=ensemble_id,
            )
            model.load()
            ensemble.append(model)

    if transformer:
        treatments = np.concatenate(ds["train"].treatments, axis=0)
        treatments = ds["train"].treatments_xfm.inverse_transform(treatments)
        treatments = np.quantile(treatments, q=np.arange(0, 1 + 1 / 32, 1 / 32),)[:-1]
    else:
        treatments = np.quantile(
            ds["train"].treatments_xfm.inverse_transform(ds["train"].treatments),
            q=np.arange(0, 1 + 1 / 32, 1 / 32),
        )[:-1]

    apos_ensemble_path = ensemble_dir / "apos_ensemble.npy"
    if not apos_ensemble_path.exists():
        capos_ensemble = ensembles.predict_capos(
            ensemble=ensemble,
            dataset=ds["test"],
            treatments=treatments,
            batch_size=1 if transformer else 20000,
        )
        apos_ensemble = capos_ensemble.mean(2)
        np.save(apos_ensemble_path, apos_ensemble)
    else:
        apos_ensemble = np.load(apos_ensemble_path)

    log_lambda = 0.05
    apo_limits_1_path = ensemble_dir / "apo_limits_0.05.npy"
    if not apo_limits_1_path.exists():
        lower_capos, upper_capos = ensembles.predict_intervals(
            ensemble=ensemble,
            dataset=ds["test"],
            treatments=treatments,
            log_lambda=log_lambda,
            num_samples=1 if transformer else 100,
            batch_size=10000,
        )
        lower_apos = np.expand_dims(lower_capos.mean(2), 0)
        upper_apos = np.expand_dims(upper_capos.mean(2), 0)
        apo_limits_1 = np.concatenate([lower_apos, upper_apos], axis=0)
        np.save(apo_limits_1_path, apo_limits_1)
    else:
        apo_limits_1 = np.load(apo_limits_1_path)

    log_lambda = 0.1
    apo_limits_2_path = ensemble_dir / "apo_limits_0.1.npy"
    if not apo_limits_2_path.exists():
        lower_capos, upper_capos = ensembles.predict_intervals(
            ensemble=ensemble,
            dataset=ds["test"],
            treatments=treatments,
            log_lambda=log_lambda,
            num_samples=1 if transformer else 100,
            batch_size=1 if transformer else 10000,
        )
        lower_apos = np.expand_dims(lower_capos.mean(2), 0)
        upper_apos = np.expand_dims(upper_capos.mean(2), 0)
        apo_limits_2 = np.concatenate([lower_apos, upper_apos], axis=0)
        np.save(apo_limits_2_path, apo_limits_2)
    else:
        apo_limits_2 = np.load(apo_limits_2_path)

    log_lambda = 0.2
    apo_limits_3_path = ensemble_dir / "apo_limits_0.2.npy"
    if not apo_limits_3_path.exists():
        lower_capos, upper_capos = ensembles.predict_intervals(
            ensemble=ensemble,
            dataset=ds["test"],
            treatments=treatments,
            log_lambda=log_lambda,
            num_samples=100,
            batch_size=10000,
        )
        lower_apos = np.expand_dims(lower_capos.mean(2), 0)
        upper_apos = np.expand_dims(upper_capos.mean(2), 0)
        apo_limits_3 = np.concatenate([lower_apos, upper_apos], axis=0)
        np.save(apo_limits_3_path, apo_limits_3)
    else:
        apo_limits_3 = np.load(apo_limits_3_path)

    ds_dict[k] = ds
    treatments_dict[k] = treatments
    ensemble_dict[k] = ensemble
    apo_ensemble_dict[k] = apos_ensemble
    apo_limits_1_dict[k] = apo_limits_1
    apo_limits_2_dict[k] = apo_limits_2
    apo_limits_3_dict[k] = apo_limits_3
    target_keys_dict[k] = target_keys

alpha = 0.05

_, ax = plt.subplots(2, 2, figsize=(12, 12))
for idx_outcome in range(len(target_keys_dict[comparison_key])):
    i, j = idx_outcome // 2, idx_outcome % 2
    for k in experiment_dict.keys():
        scaler = MinMaxScaler()
        scaler = scaler.fit(apo_ensemble_dict[k][idx_outcome].mean(0).reshape(-1, 1))
        _ = ax[i][j].plot(
            treatments_dict[k],
            scaler.transform(apo_ensemble_dict[k][idx_outcome].mean(0).reshape(-1, 1)),
            label=k,
        )
        if k == comparison_key:
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                label=r"$\Lambda \to 1.0 $",
            )
    _ = ax[i][j].legend(title=r"$\alpha=$" + f"{alpha}", loc="upper left",)
    _ = ax[i][j].set_xlabel(ds["train"].treatment_names[0])
    _ = ax[i][j].set_ylabel(target_keys_dict[comparison_key][idx_outcome])
plt.savefig(f"{comparison_name}-1.pdf")
plt.show()

_, ax = plt.subplots(2, 2, figsize=(12, 12))
for idx_outcome in range(len(target_keys_dict[comparison_key])):
    i, j = idx_outcome // 2, idx_outcome % 2
    for k in experiment_dict.keys():
        scaler = MinMaxScaler()
        scaler = scaler.fit(apo_ensemble_dict[k][idx_outcome].mean(0).reshape(-1, 1))
        _ = ax[i][j].plot(
            treatments_dict[k],
            scaler.transform(apo_ensemble_dict[k][idx_outcome].mean(0).reshape(-1, 1)),
            label=k,
        )
        if k == comparison_key:
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                label=r"$\Lambda \to 1.0 $",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_1_dict[k][1][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C1",
                label=r"$\Lambda=$" + f"{np.exp(0.1):.01f}",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_1_dict[k][0][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C1",
            )
    _ = ax[i][j].legend(title=r"$\alpha=$" + f"{alpha}", loc="upper left",)
    _ = ax[i][j].set_xlabel(ds["train"].treatment_names[0])
    _ = ax[i][j].set_ylabel(target_keys_dict[comparison_key][idx_outcome])
plt.savefig(f"{comparison_name}-1.05.pdf")
plt.show()

_, ax = plt.subplots(2, 2, figsize=(12, 12))
for idx_outcome in range(len(target_keys_dict[comparison_key])):
    i, j = idx_outcome // 2, idx_outcome % 2
    for k in experiment_dict.keys():
        scaler = MinMaxScaler()
        scaler = scaler.fit(apo_ensemble_dict[k][idx_outcome].mean(0).reshape(-1, 1))
        _ = ax[i][j].plot(
            treatments_dict[k],
            scaler.transform(apo_ensemble_dict[k][idx_outcome].mean(0).reshape(-1, 1)),
            label=k,
        )
        if k == comparison_key:
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                label=r"$\Lambda \to 1.0 $",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_1_dict[k][1][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C1",
                label=r"$\Lambda=$" + f"{np.exp(0.1):.01f}",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_1_dict[k][0][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C1",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_2_dict[k][1][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_limits_1_dict[k][1][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C2",
                label=r"$\Lambda=$" + f"{np.exp(0.2):.01f}",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_2_dict[k][0][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_limits_1_dict[k][0][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C2",
            )
    _ = ax[i][j].legend(title=r"$\alpha=$" + f"{alpha}", loc="upper left",)
    _ = ax[i][j].set_xlabel(ds["train"].treatment_names[0])
    _ = ax[i][j].set_ylabel(target_keys_dict[comparison_key][idx_outcome])
plt.savefig(f"{comparison_name}-1.1.pdf")
plt.show()

_, ax = plt.subplots(2, 2, figsize=(12, 12))
for idx_outcome in range(len(target_keys_dict[comparison_key])):
    i, j = idx_outcome // 2, idx_outcome % 2
    for k in experiment_dict.keys():
        scaler = MinMaxScaler()
        scaler = scaler.fit(apo_ensemble_dict[k][idx_outcome].mean(0).reshape(-1, 1))
        _ = ax[i][j].plot(
            treatments_dict[k],
            scaler.transform(apo_ensemble_dict[k][idx_outcome].mean(0).reshape(-1, 1)),
            label=k,
        )
        if k == comparison_key:
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                label=r"$\Lambda \to 1.0 $",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_1_dict[k][1][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C1",
                label=r"$\Lambda=$" + f"{np.exp(0.1):.01f}",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_1_dict[k][0][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_ensemble_dict[k][idx_outcome], alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C1",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_2_dict[k][1][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_limits_1_dict[k][1][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C2",
                label=r"$\Lambda=$" + f"{np.exp(0.2):.01f}",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_2_dict[k][0][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_limits_1_dict[k][0][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C2",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_3_dict[k][1][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_limits_2_dict[k][1][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C3",
                label=r"$\Lambda=$" + f"{np.exp(0.5):.01f}",
            )
            _ = ax[i][j].fill_between(
                x=treatments_dict[k],
                y1=scaler.transform(
                    np.quantile(
                        apo_limits_3_dict[k][0][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                y2=scaler.transform(
                    np.quantile(
                        apo_limits_2_dict[k][0][idx_outcome], 1 - alpha / 2, axis=0
                    ).reshape(-1, 1)
                ).flatten(),
                alpha=0.2,
                color="C3",
            )
    _ = ax[i][j].legend(title=r"$\alpha=$" + f"{alpha}", loc="upper left",)
    _ = ax[i][j].set_xlabel(ds["train"].treatment_names[0])
    _ = ax[i][j].set_ylabel(target_keys_dict[comparison_key][idx_outcome])
plt.savefig(f"{comparison_name}-1.2.pdf")
plt.show()