import json
import re
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns

from . import constants as c

try:
    import tueplots.bundles as bundles
except Exception:
    bundles = None

def _use_tex(value=None):
    if value is not None:
        return bool(value)
    return shutil.which("latex") is not None

def _serif_fonts(use_tex=None):
    if _use_tex(use_tex):
        return ["Computer Modern Roman"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    if "Computer Modern Roman" in available:
        return ["Computer Modern Roman"]
    return ["DejaVu Serif"]

def _style_params(use_tex=None, x_tick_size=10):
    return {
        "text.usetex": _use_tex(use_tex),
        "font.family": "serif",
        "font.serif": _serif_fonts(use_tex),
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": x_tick_size,
        "ytick.labelsize": 8,
    }

def _iclr_context(use_tex=None):
    if bundles is None:
        return {}
    try:
        ctx = bundles.iclr2024(usetex=_use_tex(use_tex))
    except TypeError:
        ctx = bundles.iclr2024()
        ctx["text.usetex"] = _use_tex(use_tex)
    return ctx

def _safe_stem(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

def save_figure_bundle(fig, base_path, dpi=600):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "svg", "pdf"):
        path = base_path.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, format=ext, **kwargs)
        paths[ext] = path
    return paths

def save_data_bundle(frame, base_path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base_path.with_suffix(".csv")
    json_path = base_path.with_suffix(".json")
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(frame.to_dict(orient="records"), indent=2, default=str) + "\n")
    return {"csv": csv_path, "json": json_path}

def col_idx_to_name(col_idx):
    col_idx_ = col_idx + len(c.UNINFORMATIVE)
    return " ".join(word.capitalize() for word in c.COLS[col_idx_].split("_"))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def scale_signal(signal, feature_idx, scaler_params, apply_sigmoid_round=False):
    signal = np.array(signal, copy=True)
    valid_mask = ~np.isnan(signal)

    if feature_idx in c.CONTINUOUS_REAL_VALUED_IDX:
        real_idx = c.CONTINUOUS_REAL_VALUED_IDX.index(feature_idx)
        signal[valid_mask] *= scaler_params["real_scale"][real_idx]
        signal[valid_mask] += scaler_params["real_center"][real_idx]
    elif feature_idx in c.CONTINUOUS_POSITIVE_IDX:
        pos_idx = c.CONTINUOUS_POSITIVE_IDX.index(feature_idx)
        signal[valid_mask] *= scaler_params["positive_scale"][pos_idx]
        signal[valid_mask] += scaler_params["positive_center"][pos_idx]
        signal[valid_mask] = np.expm1(signal[valid_mask])
    elif feature_idx in c.BINARY_IDX and apply_sigmoid_round:
        signal[valid_mask] = np.round(sigmoid(signal[valid_mask]))

    return signal

def clip_prediction_for_plot(signal, feature_idx):
    signal = np.array(signal, copy=True)
    feature_name = c.FEATURE_COLS[feature_idx]
    bounds = c.CLIP_INFO.get(feature_name)
    if bounds is None:
        return signal

    min_val, max_val = bounds
    valid_mask = ~np.isnan(signal)
    if min_val is not None:
        signal[valid_mask] = np.maximum(signal[valid_mask], min_val)
    if max_val is not None:
        signal[valid_mask] = np.minimum(signal[valid_mask], max_val)
    return signal

def highlight_missing_values(ax, original_plot, mask_plot):
    original_missing_where = mask_plot == 0
    artificial_missing_where = mask_plot == 2

    if np.all(original_missing_where):
        ax.fill_between(
            range(len(original_plot)), ax.get_ylim()[0], ax.get_ylim()[1],
            color="grey", alpha=0.5, label="Missing (All Original)"
        )
        return

    if not (np.any(original_missing_where) or np.any(artificial_missing_where)):
        return

    valid_values = original_plot[~np.isnan(original_plot)]
    if valid_values.size > 0:
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
    else:
        min_val = 0
        max_val = 1

    if np.any(original_missing_where):
        ax.fill_between(
            range(len(original_plot)), min_val, max_val,
            where=original_missing_where, color="grey", alpha=0.3,
            label="Missing (Original)"
        )
    if np.any(artificial_missing_where):
        ax.fill_between(
            range(len(original_plot)), min_val, max_val,
            where=artificial_missing_where, color="purple", alpha=0.3,
            label="Missing (Artificial)"
        )

def reconstruction_frame(original_plot, mask_plot, reconstruction_plots, model_names, patient_id, feature_idx):
    frame = pd.DataFrame({
        "patient_id": patient_id,
        "feature": col_idx_to_name(feature_idx),
        "time_point": np.arange(len(original_plot)),
        "original": original_plot,
        "mask": mask_plot.astype(int),
    })
    for model_name, rec_plot in zip(model_names, reconstruction_plots):
        frame[f"reconstruction_{model_name}"] = rec_plot
    return frame

def plot_signals(
    original,
    reconstructions,
    masks,
    lengths,
    users,
    feature_idx,
    model_names,
    num_samples=15,
    output_dir="plots",
    writer=None,
    scaler_params=None,
    selected_test=True,
    use_tex=None,
):
    plt.rcParams.update(_style_params(use_tex))
    colors = {"a0": "#FF0000", "a1": "#00A651", "a2": "#0056FF"}
    saved = []

    if selected_test:
        iterator = enumerate(users)
    else:
        iterator = ((i, i) for i in range(min(num_samples, len(users))))

    for i, user in iterator:
        length = int(lengths[i]) if selected_test else original.shape[-1]
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        original_plot = np.copy(original[i, feature_idx, :length])
        mask_plot = masks[i, feature_idx, :length]

        if not selected_test:
            original_plot[mask_plot == 0] = np.nan

        original_plot = scale_signal(original_plot, feature_idx, scaler_params, apply_sigmoid_round=False)
        ax.plot(np.arange(length), original_plot, label="Original", color="black", linewidth=1.3)

        reconstruction_plots = []
        for j, rec in enumerate(reconstructions):
            model_name = model_names[j]
            rec_plot = np.copy(rec[i, feature_idx, :length])
            rec_plot = scale_signal(rec_plot, feature_idx, scaler_params, apply_sigmoid_round=True)
            rec_plot = clip_prediction_for_plot(rec_plot, feature_idx)
            reconstruction_plots.append(rec_plot)
            ax.plot(np.arange(length), rec_plot, label=f"Model {model_name.upper()}", color=colors[model_name], linewidth=1.0)

        highlight_missing_values(ax, original_plot, mask_plot)
        ax.set_xlabel("Time point")
        ax.set_ylabel(col_idx_to_name(feature_idx))
        ax.legend(loc="best", frameon=False, ncol=2)
        fig.tight_layout()

        stem = f"patient_{int(user)}_feature_{_safe_stem(col_idx_to_name(feature_idx))}"
        base_path = Path(output_dir) / stem
        frame = reconstruction_frame(original_plot, mask_plot, reconstruction_plots, model_names, int(user), feature_idx)
        figure_paths = save_figure_bundle(fig, base_path, dpi=600)
        data_paths = save_data_bundle(frame, base_path)
        if writer is not None:
            writer.add_figure(f"Single/Patient_{user}/Feature_{col_idx_to_name(feature_idx)}", fig)
        plt.close(fig)
        saved.append({"figures": figure_paths, "data": data_paths})

    return saved

def embedding_probability_frame(indices, patient_id, model_type, length=None, top_n=5):
    indices = np.asarray(indices)
    if length is not None:
        indices = indices[: int(length)]
    indices = indices.flatten()
    total = int(indices.size)
    if total == 0:
        raise ValueError("Cannot plot embedding probabilities for an empty sequence.")

    counter = Counter(int(x) for x in indices)
    most_common = counter.most_common(top_n)
    rows = []
    used = set()
    for rank, (embedding_id, count) in enumerate(most_common, start=1):
        used.add(embedding_id)
        rows.append({
            "patient_id": int(patient_id),
            "model": str(model_type),
            "rank": rank,
            "embedding_id": int(embedding_id),
            "label": str(int(embedding_id)),
            "count": int(count),
            "probability": float(count / total),
            "is_others": False,
            "total": total,
        })

    others_count = sum(count for embedding_id, count in counter.items() if embedding_id not in used)
    rows.append({
        "patient_id": int(patient_id),
        "model": str(model_type),
        "rank": len(rows) + 1,
        "embedding_id": None,
        "label": "Others",
        "count": int(others_count),
        "probability": float(others_count / total),
        "is_others": True,
        "total": total,
    })
    return pd.DataFrame(rows)

def plot_embedding_probabilities(records, output_dir=".", filename_prefix="embedding_probabilities", top_n=5, use_tex=None):
    frames = []
    for record in records:
        frames.append(
            embedding_probability_frame(
                record["indices"],
                record["patient_id"],
                record["model"],
                length=record.get("length"),
                top_n=record.get("top_n", top_n),
            )
        )
    data = pd.concat(frames, ignore_index=True)

    with plt.rc_context(_iclr_context(use_tex)):
        plt.rcParams.update(_style_params(use_tex, x_tick_size=8))
        n_panels = len(frames)
        fig, axes = plt.subplots(n_panels, 1, figsize=(3.6, 1.55 * n_panels), squeeze=False)
        axes = axes[:, 0]
        for ax, frame in zip(axes, frames):
            palette = sns.color_palette("viridis", len(frame))
            sns.barplot(x="label", y="probability", hue="label", data=frame, palette=palette, legend=False, ax=ax)
            ax.set_ylim(0, min(1.0, max(0.1, frame["probability"].max() * 1.18)))
            ax.set_ylabel("Probability")
            ax.set_xlabel("Day types")
            ax.tick_params(axis="x", rotation=0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

        base_path = Path(output_dir) / filename_prefix
        figure_paths = save_figure_bundle(fig, base_path, dpi=600)
        data_paths = save_data_bundle(data, base_path)
        plt.close(fig)
        return {"figures": figure_paths, "data": data_paths}

def plot_embedding_probability(indices, patient_id, model_type, length=None, top_n=5, output_dir=".", use_tex=None):
    return plot_embedding_probabilities(
        [{"indices": indices, "patient_id": patient_id, "model": model_type, "length": length, "top_n": top_n}],
        output_dir=output_dir,
        filename_prefix=f"embedding_probabilities_{model_type}_top_{top_n}_patient_{int(patient_id)}",
        top_n=top_n,
        use_tex=use_tex,
    )
