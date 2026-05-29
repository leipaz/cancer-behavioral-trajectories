import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import constants as c

def load_scaler_params(path):
    data = np.load(path)
    return {key: data[key] for key in data.files}

def replace_out_of_bounds_with_nan(df):
    df = df.copy()
    for col, (min_val, max_val) in c.CLIP_INFO.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if min_val is not None:
            df[col] = df[col].where(df[col] >= min_val, np.nan)
        if max_val is not None:
            df[col] = df[col].where(df[col] <= max_val, np.nan)
    return df

def load_daily_summary(path):
    df = pd.read_csv(path)
    missing = sorted(set(c.COLS) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    df = df[c.COLS].copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = replace_out_of_bounds_with_nan(df)
    return df.sort_values(["user", "date_time"]).reset_index(drop=True)

def aggregate_duplicate_user_days(df):
    key_cols = ["user", "date_time"]
    duplicate_mask = df.duplicated(subset=key_cols, keep=False)
    if not duplicate_mask.any():
        return df

    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    singletons = df.loc[~duplicate_mask]
    duplicates = df.loc[duplicate_mask]
    binary_cols = set(c.BINARY_COLS)

    def agg_binary(series):
        values = pd.to_numeric(series, errors="coerce").dropna()
        return np.nan if values.empty else np.uint8(int(values.max()))

    def agg_continuous(series):
        values = pd.to_numeric(series, errors="coerce")
        return np.nan if values.notna().sum() == 0 else float(values.mean(skipna=True))

    def agg_service(series):
        values = series.dropna()
        if values.empty:
            return np.nan
        mode = values.mode()
        return mode.iloc[0] if not mode.empty else values.iloc[0]

    agg_map = {}
    for col in df.columns:
        if col in key_cols:
            continue
        if col == "service":
            agg_map[col] = agg_service
        elif col in binary_cols:
            agg_map[col] = agg_binary
        else:
            agg_map[col] = agg_continuous

    aggregated = duplicates.groupby(key_cols, as_index=False).agg(agg_map)
    out = (
        pd.concat([singletons, aggregated], ignore_index=True)
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    assert not out.duplicated(subset=key_cols).any()
    return out

def build_selected_test_sequences(df):
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d")
    df = df.sort_values(by="date_time")
    df = aggregate_duplicate_user_days(df)

    sequences = []
    for user_id, sample in df.groupby("user", sort=False):
        if sample["date_time"].duplicated().any():
            continue
        full_range = pd.date_range(sample["date_time"].min(), sample["date_time"].max(), freq="D")
        sample = sample.set_index("date_time").reindex(full_range).reset_index()
        sample = sample.rename(columns={"index": "date_time"}).fillna(np.nan)
        sample["user"] = user_id
        sequences.append(sample)

    if not sequences:
        raise ValueError("No patient sequence remains after selected-test preprocessing.")

    out = pd.concat(sequences).dropna(subset=["user"])
    out["weekend"] = (out["date_time"].dt.dayofweek >= 5).astype(np.uint8)
    return out

def apply_scaler(df, scaler_params):
    df = df.copy()
    df[c.CONTINUOUS_POSITIVE_COLS] = np.log1p(df[c.CONTINUOUS_POSITIVE_COLS])
    df[c.CONTINUOUS_POSITIVE_COLS] = (
        df[c.CONTINUOUS_POSITIVE_COLS] - scaler_params["positive_center"]
    ) / scaler_params["positive_scale"]
    df[c.CONTINUOUS_REAL_VALUED_COLS] = (
        df[c.CONTINUOUS_REAL_VALUED_COLS] - scaler_params["real_center"]
    ) / scaler_params["real_scale"]
    return df

class DailySummaryDataset(Dataset):
    def __init__(
        self,
        data,
        scaler_params,
        missingness_mode=c.DEFAULT_MISSINGNESS_MODE,
        missing_rate=c.DEFAULT_MISSING_RATE,
    ):
        self.dataset = build_selected_test_sequences(data)
        self.dataset = apply_scaler(self.dataset, scaler_params)
        self.indices = pd.unique(self.dataset["user"]).tolist()
        if missingness_mode not in {"none", "NONE", None}:
            raise ValueError("VQ-VAE inference uses observed missingness only.")
        self.missingness_mode = "none"
        self.missing_rate = 0.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        patient_id = self.indices[index]
        sample = self.dataset[self.dataset["user"] == patient_id].reset_index(drop=True)
        dates = sample["date_time"]
        signal_df = sample.drop(c.UNINFORMATIVE, axis=1)
        mask_signal_df = 1 - signal_df.isna()
        some_observed = np.count_nonzero(mask_signal_df, axis=1)
        some_observed[some_observed != 0] = 1

        signal = signal_df.to_numpy()
        mask_signal = mask_signal_df.to_numpy(dtype=np.uint8)
        mask = mask_signal.copy()

        signal_imp = copy.deepcopy(signal)
        signal_imp[mask_signal == 0] = 0

        sample = {
            "input": {
                "signal": torch.from_numpy(signal.transpose((1, 0))),
                "signal_imp": torch.from_numpy(signal_imp.transpose((1, 0))),
                "mask_signal": torch.from_numpy(mask_signal.transpose((1, 0))),
                "mask": torch.from_numpy(mask.transpose((1, 0))),
            },
            "future": {
                "signal": None,
                "signal_imp": None,
                "mask_signal": None,
                "mask": None,
            },
            "length": signal.shape[0],
            "user": patient_id,
            "dates": dates,
            "some_observed": some_observed,
        }
        return sample

def collate_full_sequences(batch):
    input_batch = [item["input"] for item in batch]
    future_batch = [item["future"] for item in batch]

    def collate_and_pad(group):
        collated = {}
        for key in group[0].keys():
            data = [item[key] for item in group if item[key] is not None and item[key].size(1) > 0]
            if data:
                max_len = max(d.size(1) for d in data)
                data_padded = [torch.nn.functional.pad(d, (0, max_len - d.size(1))) for d in data]
                collated[key] = torch.stack(data_padded, dim=0)
            else:
                collated[key] = None
        return collated

    return {
        "input": collate_and_pad(input_batch),
        "future": collate_and_pad(future_batch),
        "lengths": torch.tensor([item["length"] for item in batch]),
        "users": torch.tensor([int(item["user"]) for item in batch]),
        "dates": [item["dates"] for item in batch],
        "some_observed": [item["some_observed"] for item in batch],
    }
