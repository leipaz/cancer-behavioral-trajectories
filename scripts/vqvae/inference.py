import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import constants as c
from .data import DailySummaryDataset, collate_full_sequences, load_daily_summary, load_scaler_params
from .model import VQVAE

def set_reproducible(seed):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def save_profiles(profiles, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(profiles, handle)
    return output_path

def load_models(models_dir, modes=c.DEFAULT_MODES, device="cpu"):
    models = {}
    models_dir = Path(models_dir)
    config = dict(c.MODEL_CONFIG)
    config["p"] = config.pop("dropout")
    for mode in modes:
        checkpoint = models_dir / f"vqvae_{mode}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = VQVAE(mask_flag=int(mode[-1]), **config)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        models[mode] = model.to(device).eval()
    return models

def make_loader(
    data_csv,
    scaler_path,
    batch_size=c.DEFAULT_BATCH_SIZE,
    missingness_mode=c.DEFAULT_MISSINGNESS_MODE,
    missing_rate=c.DEFAULT_MISSING_RATE,
):
    data = load_daily_summary(data_csv)
    scaler = load_scaler_params(scaler_path)
    dataset = DailySummaryDataset(
        data,
        scaler,
        missingness_mode=missingness_mode,
        missing_rate=missing_rate,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_full_sequences, num_workers=0)
    return loader, dataset

def _top_info_for_sequence(inds, seq, orig_info, length, n):
    most = Counter(seq).most_common(n)
    top_ids = {idx for idx, _ in most}
    mapped = np.where(np.isin(inds, list(top_ids)), inds, -1)
    top_info = np.empty(length, dtype=orig_info.dtype)

    for t in range(length):
        row = orig_info[t]
        mask_top = np.isin(row["embed_id"], list(top_ids))
        in_top = row[mask_top]
        out_top = row[~mask_top]
        nearest = out_top[np.argmin(out_top["eu_dist"])]
        others = np.array(
            [
                (
                    nearest["rank"],
                    -1,
                    np.mean(out_top["eu_dist"]),
                    1.0 - in_top["pseudo_probs"].sum(),
                )
            ],
            dtype=row.dtype,
        )
        top_info[t] = np.concatenate((in_top, others))

    return mapped, top_info

def generate_profiles(
    data_csv=c.DEFAULT_DATA_CSV,
    models_dir=c.DEFAULT_MODELS_DIR,
    scaler_path=c.DEFAULT_SCALER,
    output_path=c.DEFAULT_OUTPUT,
    modes=c.DEFAULT_MODES,
    top_n_values=c.DEFAULT_TOP_N,
    batch_size=c.DEFAULT_BATCH_SIZE,
    device="cpu",
    missingness_mode=c.DEFAULT_MISSINGNESS_MODE,
    missing_rate=c.DEFAULT_MISSING_RATE,
    seed=c.DEFAULT_SEED,
    return_reconstructions=False,
):
    set_reproducible(seed)

    oncology_loader, dataset = make_loader(data_csv, scaler_path, batch_size, missingness_mode, missing_rate)
    models = load_models(models_dir, modes, device)

    max_len = 0
    for batch in oncology_loader:
        max_len = max(max_len, int(batch["lengths"].max().item()))

    embedding_counts = {mode: {n: {} for n in top_n_values} for mode in modes}
    all_original_signals = []
    all_masks = []
    all_lengths = []
    all_users = []
    all_dates = []
    all_some_observed = []
    all_reconstructions = {mode: [] for mode in modes}

    for batch in oncology_loader:
        inp = batch["input"]["signal_imp"].to(device).float()
        original = batch["input"]["signal"].to(device).float()
        msk = batch["input"]["mask_signal"].to(device).float()
        lengths = batch["lengths"].cpu().numpy()
        users = batch["users"].cpu().numpy()
        dates = batch["dates"]
        some_obs = batch["some_observed"]

        inp = torch.nn.functional.pad(inp, (0, max_len - inp.size(2)))
        original = torch.nn.functional.pad(original, (0, max_len - original.size(2)))
        msk = torch.nn.functional.pad(msk, (0, max_len - msk.size(2)))
        msk_proc = msk.clone()
        msk_proc[msk == 2] = 0
        inp[(msk == 0) | (msk == 2)] = 0

        if return_reconstructions:
            all_original_signals.append(original.cpu().numpy())
            all_masks.append(msk.cpu().numpy())
            all_lengths.extend(int(x) for x in lengths)
            all_users.extend(int(x) for x in users)
            all_dates.extend(dates)
            all_some_observed.extend(some_obs)

        for mode, model in models.items():
            with torch.no_grad():
                recons, _, indices, embed_info = model(inp, msk_proc)
            inds = indices.cpu().numpy()

            if return_reconstructions:
                all_reconstructions[mode].append(recons.detach().cpu().numpy())

            for i, uid in enumerate(users):
                length = int(lengths[i])
                seq = inds[i, :length]
                orig_info = embed_info[i, :length]
                for n in top_n_values:
                    mapped, top_info = _top_info_for_sequence(inds[i], seq, orig_info, length, n)
                    embedding_counts[mode][n].setdefault(int(uid), []).append(
                        (
                            length,
                            seq,
                            mapped,
                            dates[i],
                            some_obs[i],
                            orig_info,
                            top_info,
                        )
                    )

    if output_path is not None:
        save_profiles(embedding_counts, output_path)

    if return_reconstructions:
        plot_data = {
            "original": np.concatenate(all_original_signals, axis=0),
            "masks": np.concatenate(all_masks, axis=0),
            "lengths": np.array(all_lengths),
            "users": np.array(all_users),
            "dates": all_dates,
            "some_observed": all_some_observed,
            "reconstructions": [np.concatenate(all_reconstructions[mode], axis=0) for mode in modes],
            "model_names": list(modes),
        }
        return embedding_counts, plot_data, dataset
    return embedding_counts

def inverse_transform_signal(signal, scaler_params):
    signal = np.array(signal, copy=True)
    signal[c.CONTINUOUS_REAL_VALUED_IDX, :] *= scaler_params["real_scale"][:, None]
    signal[c.CONTINUOUS_REAL_VALUED_IDX, :] += scaler_params["real_center"][:, None]
    signal[list(c.CONTINUOUS_POSITIVE_IDX), :] *= scaler_params["positive_scale"][:, None]
    signal[list(c.CONTINUOUS_POSITIVE_IDX), :] += scaler_params["positive_center"][:, None]
    signal[list(c.CONTINUOUS_POSITIVE_IDX), :] = np.expm1(signal[list(c.CONTINUOUS_POSITIVE_IDX), :])
    return signal
