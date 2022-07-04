import numpy as np
import pandas as pd

from datetime import datetime

from torch.utils import data

from sklearn import preprocessing

# N_PIXELS = 210 # set as median of number of pixels per day for low resolution data
N_PIXELS = 8564 # set as median of number of pixels per day for high resolution data

class JASMIN(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        resolution: str,
        x_vars: list = None,
        t_var: str = None,
        y_vars: list = None,
        t_bins: int = 2,
        filter_aod: bool = True,
        filter_precip: bool = True,
        filter_lwp: bool = True,
        filter_re: bool = True,
        bootstrap=False,
    ) -> None:
        super(JASMIN, self).__init__()
        # Handle default values
        if x_vars is None:
            x_vars = (
                [
                    "MERRA_RH950",
                    "MERRA_RH850",
                    "MERRA_RH700",
                    "MERRA_LTS",
                    "MERRA_W500",
                    "ERA_sst",
                ]
                if resolution == "high"
                else ["RH900", "RH850", "RH700", "LTS", "w500", "whoi_sst", "EIS"]
            )
        if y_vars is None:
            y_vars = (
                [
                    "Cloud_Effective_Radius",
                    "Cloud_Optical_Thickness",
                    "Cloud_Water_Path",
                    "Nd",
                ]
                if resolution == "high"
                else ["l_re", "cod", "cwp", "liq_pc"]
            )
        if t_var is None: 
            t_var = "MERRA_aod" if resolution == 'high' else 'tot_aod'
        # Read csv
        df = pd.read_csv(root, index_col=0)
        # Filtering AOD, precipitation, lwp, re
        if resolution == 'low': 
            if t_var == 'tot_aod' and filter_aod:
                df = df[df[t_var].between(0.07, 1.0)]
            if filter_precip: 
                df = df[df["precip"] < 0.5]
            if filter_lwp: 
                df = df[df["cwp"] < 250]
            if filter_re: 
                df = df[df["l_re"] < 30]
        if resolution == 'high':
            if t_var == 'MERRA_aod' and filter_aod:
                df = df[df[t_var].between(0.07, 1.0)]
            if filter_precip: 
                df = df[df["imerg_precip"] < 0.5]
                df = df[df["imerg_precip_T30"] < 0.5]
                df = df[df["imerg_precip_T60"] < 0.5]
            if filter_lwp: 
                df = df[df["Cloud_Water_Path"] < 250]
            if filter_re: 
                df = df[df["Cloud_Effective_Radius"] < 30]
        # if t_var in ["tot_aod", "MERRA_aod"] and filter_aod:
        #     df = df[df[t_var].between(0.07, 1.0)]
        # if "precip" in df.columns and filter_precip:
        #     df = df[df["precip"] < 0.5]
        # if "imerg_precip" in df.columns and filter_precip:
        #     df = df[df["imerg_precip"] < 0.5]
        # if "imerg_precip_T30" in df.columns and filter_precip:
        #     df = df[df["imerg_precip_T30"] < 0.5]
        # if "imerg_precip_T60" in df.columns and filter_precip:
        #     df = df[df["imerg_precip_T60"] < 0.5]
        # Filter lwp values
        # if "cwp" in df.columns and filter_lwp: 
        #     df = df[df["cwp"] < 250]
        # if "Cloud_Water_Path" in df.columns and filter_lwp: 
        #     df = df[df["Cloud_Water_Path"] < 250]
        # # Filter re values
        # if "l_re" in df.columns and filter_re: 
        #     df = df[df["l_re"] < 30]
        # if "Cloud_Effective_Radius" in df.columns and filter_re: 
        #     df = df[df["Cloud_Effective_Radius"] < 30]
        # Make train test valid split
        if resolution == 'high': 
            df['dates'] = pd.to_datetime(df['dates']).apply(datetime.date)
        times = "timestamp" if resolution == "low" else "dates"
        days = df[times].unique()
        days_valid = set(days[5::7])
        days_test = set(days[6::7])
        days_train = set(days).difference(days_valid.union(days_test))
        # Fit preprocessing transforms
        df_train = df[df[times].isin(days_train)]
        self.data_xfm = preprocessing.StandardScaler()
        self.data_xfm.fit(df_train[x_vars].to_numpy())
        self.treatments_xfm = (
            preprocessing.KBinsDiscretizer(n_bins=t_bins, encode="onehot-dense")
            if t_bins > 1
            else preprocessing.StandardScaler()
        )
        self.treatments_xfm.fit(df_train[t_var].to_numpy().reshape(-1, 1))
        self.targets_xfm = preprocessing.StandardScaler()
        self.targets_xfm.fit(df_train[y_vars].to_numpy())
        # Split the data
        if split == "train":
            _df = df[df[times].isin(days_train)]
        elif split == "valid":
            _df = df[df[times].isin(days_valid)]
        elif split == "test":
            _df = df[df[times].isin(days_test)]
        # Set variables
        self.data = self.data_xfm.transform(_df[x_vars].to_numpy(dtype="float32"))
        self.treatments = self.treatments_xfm.transform(
            _df[t_var].to_numpy(dtype="float32").reshape(-1, 1)
        )
        self.targets = self.targets_xfm.transform(_df[y_vars].to_numpy(dtype="float32"))
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
        root: str,
        split: str,
        resolution: str,
        x_vars: list = None,
        t_var: str = None,
        y_vars: list = None,
        t_bins: int = 2,
        filter_aod: bool = True,
        filter_precip: bool = True,
        filter_lwp: bool = True,
        filter_re: bool = True, 
        pad: bool = False,
        bootstrap=False,
    ) -> None:
        super(JASMINDaily, self).__init__()
        # Handle default values
        if x_vars is None:
            x_vars = (
                [
                    "MERRA_RH950",
                    "MERRA_RH850",
                    "MERRA_RH700",
                    "MERRA_LTS",
                    "MERRA_W500",
                    "ERA_sst",
                ]
                if resolution == "high"
                else ["RH900", "RH850", "RH700", "LTS", "w500", "whoi_sst", "EIS"]
            )
        if y_vars is None:
            y_vars = (
                [
                    "Cloud_Effective_Radius",
                    "Cloud_Optical_Thickness",
                    "Cloud_Water_Path",
                    "Nd",
                ]
                if resolution == "high"
                else ["l_re", "cod", "cwp", "liq_pc"]
            )
        if t_var is None: 
            t_var = "MERRA_aod" if resolution == 'high' else 'tot_aod'
        # Read csv
        df = pd.read_csv(root, index_col=0)
        # Filtering AOD, precipitation, lwp, re
        if resolution == 'low': 
            if t_var == 'tot_aod' and filter_aod:
                df.loc[~df[t_var].between(0.07, 1.0), y_vars] = np.nan
            if filter_precip: 
                df.loc[df["precip"] >= 0.5, y_vars] = np.nan
            if filter_lwp: 
                df.loc[df["cwp"] >= 250, y_vars] = np.nan
            if filter_re: 
                df.loc[df["l_re"] >= 30, y_vars] = np.nan
        if resolution == 'high':
            if t_var == 'MERRA_aod' and filter_aod:
                df.loc[~df[t_var].between(0.07, 1.0), y_vars] = np.nan
            if filter_precip: 
                df.loc[df["imerg_precip"] >= 0.5, y_vars] = np.nan
                df.loc[df["imerg_precip_T30"] >= 0.5, y_vars] = np.nan
                df.loc[df["imerg_precip_T60"] >= 0.5, y_vars] = np.nan
            if filter_lwp: 
                df.loc[df["Cloud_Water_Path"] >= 250, y_vars] = np.nan
            if filter_re: 
                df.loc[df["Cloud_Effective_Radius"] >= 30, y_vars] = np.nan
        # # Filter AOD values
        # if t_var in ["tot_aod", "MERRA_aod"] and filter_aod:
        #     df.loc[~df[t_var].between(0.07, 1.0), y_vars] = np.nan
        # # Filter precipitation values
        # if "precip" in df.columns and filter_precip:
        #     df.loc[df["precip"] >= 0.5, y_vars] = np.nan
        # if "imerg_precip" in df.columns and filter_precip:
        #     df.loc[df["imerg_precip"] >= 0.5, y_vars] = np.nan
        # if "imerg_precip_T30" in df.columns and filter_precip:
        #     df.loc[df["imerg_precip_T30"] >= 0.5, y_vars] = np.nan
        # if "imerg_precip_T60" in df.columns and filter_precip:
        #     df.loc[df["imerg_precip_T60"] >= 0.5, y_vars] = np.nan
        # # Filter lwp values
        # if "cwp" in df.columns and filter_lwp: 
        #     df.loc[df["cwp"] >= 250, y_vars] = np.nan
        # if "Cloud_Water_Path" in df.columns and filter_lwp: 
        #     df.loc[df["Cloud_Water_Path"] >= 250, y_vars] = np.nan
        # # Filter re values
        # if "l_re" in df.columns and filter_re: 
        #     df.loc[df["l_re"] >= 30, y_vars] = np.nan
        # if "Cloud_Effective_Radius" in df.columns and filter_re: 
        #     df.loc[df["Cloud_Effective_Radius"] >= 30, y_vars] = np.nan
        # Make train test valid split
        if 'dates' in df.columns: 
            df['dates'] = pd.to_datetime(df['dates']).apply(datetime.date)
        times = "timestamp" if resolution == "low" else "dates"
        days = df[times].unique()
        days_valid = set(days[5::7])
        days_test = set(days[6::7])
        days_train = set(days).difference(days_valid.union(days_test))
        # Fit preprocessing transforms
        df_train = df[df[times].isin(days_train)]
        self.data_xfm = preprocessing.StandardScaler()
        self.data_xfm.fit(df_train[x_vars].to_numpy())
        self.treatments_xfm = (
            preprocessing.KBinsDiscretizer(n_bins=t_bins, encode="onehot-dense")
            if t_bins > 1
            else preprocessing.StandardScaler()
        )
        self.treatments_xfm.fit(df_train[t_var].to_numpy().reshape(-1, 1))
        self.targets_xfm = preprocessing.StandardScaler()
        self.targets_xfm.fit(df_train[y_vars].to_numpy())
        # Split the data
        if split == "train":
            _df = df[df[times].isin(days_train)]
        elif split == "valid":
            _df = df[df[times].isin(days_valid)]
        elif split == "test":
            _df = df[df[times].isin(days_test)]
        _df = _df.groupby(times)
        # Set variables
        self.data = []
        self.treatments = []
        self.targets = []
        self.position = []
        for _, group in _df:
            if len(group) > 1:
                targets = self.targets_xfm.transform(group[y_vars].to_numpy(dtype="float32"))
                if np.isnan(targets).sum() != (lambda x: x[0]*x[1])(targets.shape): # target is not full of nan
                    self.data.append(
                        self.data_xfm.transform(group[x_vars].to_numpy(dtype="float32"))
                    )
                    self.treatments.append(
                        self.treatments_xfm.transform(
                            group[t_var].to_numpy(dtype="float32").reshape(-1, 1)
                        )
                    )
                    self.targets.append(
                        self.targets_xfm.transform(group[y_vars].to_numpy(dtype="float32"))
                    )
                    self.position.append(group[["lats", "lons"]].to_numpy(dtype="float32"))
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
        root_lr: str,
        root_hr: str,
        split: str,
        x_vars: list = None,
        t_var: str = None,
        y_vars: list = None,
        t_bins: int = 2,
        filter_aod: bool = True,
        filter_precip: bool = True,
        filter_lwp: bool = True,
        filter_re: bool = True, 
        pad: bool = False,
        bootstrap=False,
    ) -> None:
        super(JASMINCombo, self).__init__()
        # Handle default values
        if x_vars is None:
            x_vars = [
                "MERRA_RH950",
                "MERRA_RH850",
                "MERRA_RH700",
                "MERRA_LTS",
                "MERRA_W500",
                "ERA_sst",
            ]
        if y_vars is None:
            y_vars = ["l_re", "cod", "cwp", "liq_pc"]
        if t_var is None: 
            t_var = 'tot_aod'
        # Read csv
        hr = pd.read_csv(root_hr, index_col=0)
        lr = pd.read_csv(root_lr, index_col=0)
        # Filtering AOD, precipitation, lwp, re
        if t_var == 'tot_aod' and filter_aod:
            lr.loc[~lr[t_var].between(0.07, 1.0), y_vars] = np.nan
        if filter_precip: 
            lr.loc[lr["precip"] >= 0.5, y_vars] = np.nan
        if filter_lwp: 
            lr.loc[lr["cwp"] >= 250, y_vars] = np.nan
        if filter_re: 
            lr.loc[lr["l_re"] >= 30, y_vars] = np.nan
        # Make train test valid split
        hr['dates'] = pd.to_datetime(hr['dates'])
        hr['dates_only'] = hr['dates'].apply(datetime.date)
        hr['hours_only'] = hr['dates'].apply(datetime.time).apply(lambda x: x.hour)
        lr['dates_only'] = lr['timestamp'].apply(datetime.fromtimestamp)
        days = hr['dates_only'].unique()
        days_valid = set(days[5::7])
        days_test = set(days[6::7])
        days_train = set(days).difference(days_valid.union(days_test))
        # Fit preprocessing transforms
        lr_df_train = lr[lr['dates_only'].isin(days_train)]
        hr_df_train = hr[hr['dates_only'].isin(days_train)]
        self.data_xfm = preprocessing.StandardScaler()
        self.data_xfm.fit(hr_df_train[x_vars].to_numpy())
        self.treatments_xfm = (
            preprocessing.KBinsDiscretizer(n_bins=t_bins, encode="onehot-dense")
            if t_bins > 1
            else preprocessing.StandardScaler()
        )
        self.treatments_xfm.fit(lr_df_train[t_var].to_numpy().reshape(-1, 1))
        self.targets_xfm = preprocessing.StandardScaler()
        self.targets_xfm.fit(lr_df_train[y_vars].to_numpy())
        # Split the data
        if split == "train":
            _lr = lr[lr['dates_only'].isin(days_train)]
            _hr = hr[hr['dates_only'].isin(days_train)]
        elif split == "valid":
            _lr = lr[lr['dates_only'].isin(days_valid)]
            _hr = hr[hr['dates_only'].isin(days_valid)]
        elif split == "test":
            _lr = lr[lr['dates_only'].isin(days_test)]
            _hr = hr[hr['dates_only'].isin(days_test)]        
        _lr = _lr.groupby('dates_only')
        _hr = _hr.groupby('dates_only')
        # Set variables
        self.data = []
        self.treatments = []
        self.targets = []
        self.position = []
        for _, group in _hr:
            self.data.append(
                self.data_xfm.transform(group[x_vars].to_numpy(dtype="float32"))
            )
            self.position.append(group[["lats", "lons", "hours_only"]].to_numpy(dtype="float32"))
        for _, group in _lr:
            self.treatments.append(
                self.treatments_xfm.transform(
                    group[t_var].to_numpy(dtype="float32").reshape(-1, 1)
                )
            )
            self.targets.append(
                self.targets_xfm.transform(group[y_vars].to_numpy(dtype="float32"))
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
        return covariates, treatments, targets, position