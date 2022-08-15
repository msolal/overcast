import numpy as np
import pandas as pd

from datetime import datetime

from torch.utils import data

from sklearn import preprocessing

import random

RANDOM_SEED = 42
N_PIXELS = 210 # median number of pixels per day for low resolution data
# N_PIXELS = 8564 # median number of pixels per day for high resolution data


class JASMIN(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset: str,
        split: str,
        x_vars: list = ["RH900", "RH850", "RH700", "EIS", "LTS", "W500", "SST"],
        t_var: str = "AOD",
        y_vars: list = ["re", "COD", "CWP", "LPC"],
        t_bins: int = 2,
        filter_aod: bool = True,
        filter_precip: bool = True,
        filter_cwp: bool = True,
        filter_re: bool = True,
        bootstrap=False,
    ) -> None:
        super(JASMIN, self).__init__()
        # Convert variable names into their keys in the dataset
        vars_names = pd.read_csv(
            f"{data_dir}/variables.csv", index_col=0, header=0, na_filter=False
        ).loc[dataset]
        # x_vars_ds = list(filter(None, list(vars_names[x] for x in x_vars)))
        x_vars_ds = list(vars_names[x] for x in x_vars)
        t_var_ds = vars_names[t_var]
        # y_vars_ds = list(filter(None, list(vars_names[y] for y in y_vars)))
        y_vars_ds = list(vars_names[y] for y in y_vars)
        # Read csv
        df = pd.read_csv(f"{data_dir}/{dataset}.csv", index_col=0)
        # Filtering AOD
        if t_var == "AOD" and filter_aod:
            df = df[df[t_var_ds].between(0.03, 0.3)]
        # Filter precipitation
        if filter_precip:
            precip_vars = list(
                filter(
                    None,
                    list(
                        vars_names[p]
                        for p in [
                            "precipitation",
                            "precipitation_T30",
                            "precipitation_T60",
                        ]
                    ),
                )
            )
            for precip_var in precip_vars:
                df = df[df[precip_var] < 0.5]
        # Filter lwp
        if filter_cwp:
            df = df[df[vars_names["CWP"]] < 250]
        # Filter re
        if filter_re:
            re_var = vars_names["re"]
            df = df[df[re_var] < 30]
        # Convert time stamps into separate date and time columns
        timestamp_ds = vars_names["timestamp"]
        if df.dtypes[timestamp_ds] == "float64":  # if timestamp, convert to datetime
            df[timestamp_ds] = df[timestamp_ds].apply(datetime.fromtimestamp)
        df["date"] = pd.to_datetime(df[timestamp_ds]).apply(datetime.date)
        df["time"] = pd.to_datetime(df[timestamp_ds]).apply(datetime.time)
        # Make train test valid split
        days = df["date"].unique()
        random.seed(RANDOM_SEED)
        random.shuffle(days)
        days_valid = set(days[5::7])
        days_test = set(days[6::7])
        days_train = set(days).difference(days_valid.union(days_test))
        # Fit preprocessing transforms
        df_train = df[df["date"].isin(days_train)]
        self.data_xfm = preprocessing.StandardScaler()
        self.data_xfm.fit(df_train[x_vars_ds].to_numpy())
        self.treatments_xfm = (
            preprocessing.KBinsDiscretizer(n_bins=t_bins, encode="onehot-dense")
            if t_bins > 1
            else preprocessing.StandardScaler()
        )
        self.treatments_xfm.fit(df_train[t_var_ds].to_numpy().reshape(-1, 1))
        self.targets_xfm = preprocessing.StandardScaler()
        self.targets_xfm.fit(df_train[y_vars_ds].to_numpy())
        # Split the data
        if split == "train":
            _df = df[df["date"].isin(days_train)]
        elif split == "valid":
            _df = df[df["date"].isin(days_valid)]
        elif split == "test":
            _df = df[df["date"].isin(days_test)]
        # Set variables
        self.data = self.data_xfm.transform(_df[x_vars_ds].to_numpy(dtype="float32"))
        self.treatments = self.treatments_xfm.transform(
            _df[t_var_ds].to_numpy(dtype="float32").reshape(-1, 1)
        )
        self.targets = self.targets_xfm.transform(
            _df[y_vars_ds].to_numpy(dtype="float32")
        )
        # Variable properties
        self.dim_input = self.data.shape[-1]
        self.dim_targets = self.targets.shape[-1]
        self.dim_treatments = t_bins
        self.data_names = x_vars
        self.target_names = y_vars
        if t_bins > 1:
            bin_edges = self.treatments_xfm.bin_edges_[0]
            self.treatment_names = [
                f"{t_var} [{bin_edges[i]:.03f}, {bin_edges[i+1]:.03f})"
                for i in range(t_bins)
            ]
        else:
            self.treatment_names = [t_var]
        # Bootstrap sampling
        self.sample_index = np.arange(len(self.data))
        if bootstrap:
            self.sample_index = np.random.choice(self.sample_index, size=len(self.data))
            self.data = self.data[self.sample_index]
            self.treatments = self.treatments[self.sample_index]
            self.targets = self.targets[self.sample_index]

    @property
    def data_frame(self) -> pd.DataFrame:
        data = np.hstack(
            [
                self.data_xfm.inverse_transform(self.data),
                self.treatments
                if self.dim_treatments > 1
                else self.treatments_xfm.inverse_transform(self.treatments),
                self.targets_xfm.inverse_transform(self.targets),
            ],
        )
        return pd.DataFrame(
            data=data,
            columns=self.data_names + self.treatment_names + self.target_names,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> data.dataset.T_co:
        return self.data[index], self.treatments[index], self.targets[index]


class JASMINDaily(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset: str,
        split: str,
        x_vars: list = ["RH900", "RH850", "RH700", "LTS", "W500", "SST", "EIS"],
        t_var: str = "AOD",
        y_vars: list = ["re", "COD", "CWP"],
        t_bins: int = 2,
        filter_aod: bool = True,
        filter_precip: bool = True,
        filter_cwp: bool = True,
        filter_re: bool = True,
        pad: bool = False,
        bootstrap=False,
    ) -> None:
        super(JASMINDaily, self).__init__()
        # Convert variables into their name from the dataset
        vars_names = pd.read_csv(
            f"{data_dir}/variables.csv", index_col=0, header=0, na_filter=False
        ).loc[dataset]
        x_vars_ds = list(filter(None, list(vars_names[x] for x in x_vars)))
        t_var_ds = vars_names[t_var]
        y_vars_ds = list(filter(None, list(vars_names[y] for y in y_vars)))
        # Read csv
        df = pd.read_csv(f"{data_dir}/{dataset}.csv", index_col=0)
        # df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Filtering AOD
        if t_var == "AOD" and filter_aod:
            df.loc[~df[t_var_ds].between(0.03, 0.3), y_vars_ds] = np.nan
        # Filter precipitation
        if filter_precip:
            precip_vars = list(
                filter(
                    None,
                    list(
                        vars_names[p]
                        for p in [
                            "precipitation",
                            "precipitation_T30",
                            "precipitation_T60",
                        ]
                    ),
                )
            )
            for precip_var in precip_vars:
                df.loc[df[precip_var] >= 0.5, y_vars_ds] = np.nan
        # Filter lwp
        if filter_cwp:
            df.loc[df[vars_names["CWP"]] >= 250, y_vars_ds] = np.nan
        # Filter re
        if filter_re:
            df.loc[df[vars_names["re"]] >= 30, y_vars_ds] = np.nan
        # Convert time stamps into separate date and time columns
        timestamp_ds = vars_names["timestamp"]
        if df.dtypes[timestamp_ds] == "float64":  # if timestamp, convert to datetime
            df[timestamp_ds] = df[timestamp_ds].apply(datetime.fromtimestamp)
        df["date"] = pd.to_datetime(df[timestamp_ds]).apply(datetime.date)
        df["time"] = pd.to_datetime(df[timestamp_ds]).apply(datetime.time)
        # Make train test valid split
        days = df["date"].unique()
        random.seed(RANDOM_SEED)
        random.shuffle(days)
        days_valid = set(days[5::7])
        days_test = set(days[6::7])
        days_train = set(days).difference(days_valid.union(days_test))
        # Fit preprocessing transforms
        df_train = df[df["date"].isin(days_train)]
        self.data_xfm = preprocessing.StandardScaler()
        self.data_xfm.fit(df_train[x_vars_ds].to_numpy())
        self.treatments_xfm = (
            preprocessing.KBinsDiscretizer(n_bins=t_bins, encode="onehot-dense")
            if t_bins > 1
            else preprocessing.StandardScaler()
        )
        self.treatments_xfm.fit(df_train[t_var_ds].to_numpy().reshape(-1, 1))
        self.targets_xfm = preprocessing.StandardScaler()
        self.targets_xfm.fit(df_train[y_vars_ds].to_numpy())
        # Split the data
        if split == "train":
            _df = df[df["date"].isin(days_train)]
        elif split == "valid":
            _df = df[df["date"].isin(days_valid)]
        elif split == "test":
            _df = df[df["date"].isin(days_test)]
        _df = _df.groupby("date")
        # Set variables
        self.data = []
        self.treatments = []
        self.targets = []
        self.position = []
        for _, group in _df:
            if len(group) > 1:
                targets = self.targets_xfm.transform(
                    group[y_vars_ds].to_numpy(dtype="float32")
                )
                if ~np.isnan(targets).all():  # target is not full of nan
                    self.data.append(
                        self.data_xfm.transform(
                            group[x_vars_ds].to_numpy(dtype="float32")
                        )
                    )
                    # print(self.data[-1].shape)
                    self.treatments.append(
                        self.treatments_xfm.transform(
                            group[t_var_ds].to_numpy(dtype="float32").reshape(-1, 1)
                        )
                    )
                    self.targets.append(
                        self.targets_xfm.transform(
                            group[y_vars_ds].to_numpy(dtype="float32")
                        )
                    )
                    self.position.append(
                        group[[vars_names["lats"], vars_names["lons"]]].to_numpy(
                            dtype="float32"
                        )
                    )
        # Variable properties
        self.dim_input = self.data[0].shape[-1]
        self.dim_targets = self.targets[0].shape[-1]
        self.dim_treatments = t_bins
        self.data_names = x_vars
        self.target_names = y_vars
        if t_bins > 1:
            bin_edges = self.treatments_xfm.bin_edges_[0]
            self.treatment_names = [
                f"{t_var} [{bin_edges[i]:.03f}, {bin_edges[i+1]:.03f})"
                for i in range(t_bins)
            ]
        else:
            self.treatment_names = [t_var]
        self.split = split
        self.pad = pad
        # Bootstrap sampling
        if bootstrap:
            num_days = len(self.data)
            sample_index = np.arange(num_days)
            sample_index = np.random.choice(sample_index, size=num_days)
            self.data = [self.data[j] for j in sample_index]
            self.treatments = [self.treatments[j] for j in sample_index]
            self.targets = [self.targets[j] for j in sample_index]
            self.position = [self.position[j] for j in sample_index] 

    @property
    def data_frame(self):
        data = np.hstack(
            [
                self.data_xfm.inverse_transform(np.vstack(self.data)),
                np.vstack(self.treatments)
                if self.dim_treatments > 1
                else self.treatments_xfm.inverse_transform(np.vstack(self.treatments)),
                self.targets_xfm.inverse_transform(np.vstack(self.targets)),
            ],
        )
        return pd.DataFrame(
            data=data,
            columns=self.data_names + self.treatment_names + self.target_names,
        ).dropna()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> data.dataset.T_co:
        covariates = self.data[index]
        treatments = self.treatments[index]
        targets = self.targets[index]
        position = self.position[index]
        if self.pad:
            num_samples = covariates.shape[0]
            if num_samples < N_PIXELS:
                diff = N_PIXELS - num_samples
                pading = ((0, diff), (0, 0))
                covariates = np.pad(covariates, pading, constant_values=np.nan)
                treatments = np.pad(treatments, pading, constant_values=np.nan)
                targets = np.pad(targets, pading, constant_values=np.nan)
                position = np.pad(position, pading, constant_values=np.nan)
            elif num_samples > N_PIXELS:
                sample = np.random.choice(np.arange(num_samples), N_PIXELS, replace=False)
                sample.sort()
                covariates = covariates[sample]
                treatments = treatments[sample]
                targets = targets[sample]
                position = position[sample]
        position -= position.mean(0)
        if self.split == "train" and self.pad:
            position += np.random.uniform(-10, 10, size=(1, 2))
        return covariates, treatments, targets, position


class JASMINCombo(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        lr_dataset: str,
        hr_dataset: str,
        split: str,
        x_vars: list = ["RH900", "RH850", "RH700", "LTS", "W500", "SST", "EIS"],
        t_var: str = "AOD",
        y_vars: list = ["re", "COD", "CWP"],
        t_bins: int = 2,
        filter_aod: bool = True,
        filter_precip: bool = True,
        filter_cwp: bool = True,
        filter_re: bool = True,
        pad: bool = False,
        bootstrap=False,
    ) -> None:
        super(JASMINCombo, self).__init__()
        # Convert variables into their name from the dataset
        hr_vars_names = pd.read_csv(
            f"{data_dir}/variables.csv", index_col=0, header=0, na_filter=False
        ).loc[hr_dataset]
        lr_vars_names = pd.read_csv(
            f"{data_dir}/variables.csv", index_col=0, header=0, na_filter=False
        ).loc[lr_dataset]
        x_vars_lr = list(filter(None, list(lr_vars_names[x] for x in x_vars)))
        t_var_lr = lr_vars_names[t_var]
        y_vars_lr = list(filter(None, list(lr_vars_names[y] for y in y_vars)))
        x_vars_hr = list(filter(None, list(hr_vars_names[x] for x in x_vars)))
        t_var_hr = hr_vars_names[t_var]
        y_vars_hr = list(filter(None, list(hr_vars_names[y] for y in y_vars)))
        # Read csv
        hr = pd.read_csv(f"{data_dir}/{hr_dataset}.csv", index_col=0)
        lr = pd.read_csv(f"{data_dir}/{lr_dataset}.csv", index_col=0)
        # Filtering AOD
        if t_var == "AOD" and filter_aod:
            lr.loc[~lr[t_var_lr].between(0.03, 0.3), y_vars_lr] = np.nan
            hr.loc[~hr[t_var_hr].between(0.03, 0.3), y_vars_hr] = np.nan
        # Filter precipitation
        if filter_precip:
            precip_list = [
                "precipitation",
                "precipitation_T30",
                "precipitation_T60",
            ]
            precip_vars_lr = list(
                filter(None, list(lr_vars_names[p] for p in precip_list))
            )
            for precip_var in precip_vars_lr:
                lr.loc[lr[precip_var] >= 0.5, y_vars_lr] = np.nan
            precip_vars_hr = list(
                filter(None, list(hr_vars_names[p] for p in precip_list))
            )
            for precip_var in precip_vars_hr:
                hr.loc[hr[precip_var] >= 0.5, y_vars_hr] = np.nan
        # Filter lwp
        if filter_cwp:
            lr.loc[lr[lr_vars_names["CWP"]] >= 250, y_vars_lr] = np.nan
            hr.loc[hr[hr_vars_names["CWP"]] >= 250, y_vars_hr] = np.nan
        # Filter re
        if filter_re:
            lr.loc[lr[lr_vars_names["re"]] >= 30, y_vars_lr] = np.nan
            hr.loc[hr[hr_vars_names["re"]] >= 30, y_vars_hr] = np.nan
        # Make train test valid split
        timestamp_lr = lr_vars_names["timestamp"]
        timestamp_hr = hr_vars_names["timestamp"]
        if lr.dtypes[timestamp_lr] == "float64":  # if timestamp, convert to datetime
            lr[timestamp_lr] = lr[timestamp_lr].apply(datetime.fromtimestamp)
        if hr.dtypes[timestamp_hr] == "float64":  # if timestamp, convert to datetime
            hr[timestamp_hr] = lr[timestamp_hr].apply(datetime.fromtimestamp)
        lr["date"] = pd.to_datetime(lr[timestamp_lr]).apply(datetime.date)
        hr["date"] = pd.to_datetime(hr[timestamp_hr]).apply(datetime.date)
        hr["time"] = pd.to_datetime(hr[timestamp_hr]).apply(datetime.time)
        hr["hour"] = hr["time"].apply(lambda x: x.hour)
        days = hr["date"].unique()
        days_valid = set(days[5::7])
        days_test = set(days[6::7])
        days_train = set(days).difference(days_valid.union(days_test))
        # Fit preprocessing transforms
        lr_df_train = lr[lr["date"].isin(days_train)]
        hr_df_train = hr[hr["date"].isin(days_train)]
        self.data_xfm = preprocessing.StandardScaler()
        self.data_xfm.fit(hr_df_train[x_vars_hr].to_numpy())
        self.treatments_xfm = (
            preprocessing.KBinsDiscretizer(n_bins=t_bins, encode="onehot-dense")
            if t_bins > 1
            else preprocessing.StandardScaler()
        )
        self.treatments_xfm.fit(lr_df_train[t_var_lr].to_numpy().reshape(-1, 1))
        self.targets_xfm = preprocessing.StandardScaler()
        self.targets_xfm.fit(lr_df_train[y_vars_lr].to_numpy())
        # Split the data
        if split == "train":
            _lr = lr[lr["date"].isin(days_train)]
            _hr = hr[hr["date"].isin(days_train)]
        elif split == "valid":
            _lr = lr[lr["date"].isin(days_valid)]
            _hr = hr[hr["date"].isin(days_valid)]
        elif split == "test":
            _lr = lr[lr["date"].isin(days_test)]
            _hr = hr[hr["date"].isin(days_test)]
        _lr = _lr.groupby("date")
        _hr = _hr.groupby("date")
        # Set variables
        self.data = []
        self.treatments = []
        self.targets = []
        self.position = []
        for _, group in _hr:
            self.data.append(
                self.data_xfm.transform(group[x_vars_hr].to_numpy(dtype="float32"))
            )
            self.position.append(
                group[[hr_vars_names["lats"], hr_vars_names["lons"], "hour"]].to_numpy(dtype="float32")
            )
        for _, group in _lr:
            self.treatments.append(
                self.treatments_xfm.transform(
                    group[t_var_lr].to_numpy(dtype="float32").reshape(-1, 1)
                )
            )
            self.targets.append(
                self.targets_xfm.transform(group[y_vars_lr].to_numpy(dtype="float32"))
            )
        # Variable properties
        self.dim_input = self.data[0].shape[-1]
        self.dim_targets = self.targets[0].shape[-1]
        self.dim_treatments = t_bins
        self.data_names = x_vars
        self.target_names = y_vars
        if t_bins > 1:
            bin_edges = self.treatments_xfm.bin_edges_[0]
            self.treatment_names = [
                f"{t_var} [{bin_edges[i]:.03f}, {bin_edges[i+1]:.03f})"
                for i in range(t_bins)
            ]
        else:
            self.treatment_names = [t_var]
        self.split = split
        self.pad = pad
        # Bootstrap sampling
        if bootstrap:
            num_days = len(self.data)
            sample_index = np.arange(num_days)
            sample_index = np.random.choice(sample_index, size=num_days)
            self.data = [self.data[j] for j in sample_index]
            self.treatments = [self.treatments[j] for j in sample_index]
            self.targets = [self.targets[j] for j in sample_index]
            self.position = [self.position[j] for j in sample_index]

    @property
    def data_frame(self):
        data = np.hstack(
            [
                self.data_xfm.inverse_transform(np.vstack(self.data)),
                np.vstack(self.treatments)
                if self.dim_treatments > 1
                else self.treatments_xfm.inverse_transform(np.vstack(self.treatments)),
                self.targets_xfm.inverse_transform(np.vstack(self.targets)),
            ],
        )
        return pd.DataFrame(
            data=data,
            columns=self.data_names + self.treatment_names + self.target_names,
        ).dropna()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> data.dataset.T_co:
        covariates = self.data[index]
        treatments = self.treatments[index]
        targets = self.targets[index]
        position = self.position[index]
        if self.pad:
            num_samples = covariates.shape[0]
            if num_samples < 210:
                diff = 210 - num_samples
                pading = ((0, diff), (0, 0))
                covariates = np.pad(covariates, pading, constant_values=np.nan)
                treatments = np.pad(treatments, pading, constant_values=np.nan)
                targets = np.pad(targets, pading, constant_values=np.nan)
                position = np.pad(position, pading, constant_values=np.nan)
            elif num_samples > 210:
                sample = np.random.choice(np.arange(num_samples), 210, replace=False)
                sample.sort()
                covariates = covariates[sample]
                treatments = treatments[sample]
                targets = targets[sample]
                position = position[sample]
        position -= position.mean(0)
        return covariates, treatments, targets, position
