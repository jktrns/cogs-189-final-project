from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns

from .data import EEG_CHANNELS

FIGURES_DIR = Path(__file__).parent.parent / "figures"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.constrained_layout.use": True,
        }
    )
    sns.set_palette("colorblind")


def plot_within_subject_accuracy(
    results: dict[int, dict[str, dict[str, float]]],
    save_path: Path | None = None,
) -> plt.Figure:
    setup_style()
    subjects = sorted(results.keys())
    classifier_names = list(next(iter(results.values())).keys())
    n_subjects = len(subjects)
    n_classifiers = len(classifier_names)

    figure, axis = plt.subplots(figsize=(max(12, n_subjects * 0.8), 6))
    x_positions = np.arange(n_subjects)
    width = 0.8 / n_classifiers
    for index, classifier_name in enumerate(classifier_names):
        accuracies = [
            results[subject][classifier_name]["accuracy"] for subject in subjects
        ]
        deviations = [
            results[subject][classifier_name]["accuracy_std"] for subject in subjects
        ]
        axis.bar(
            x_positions + (index - n_classifiers / 2 + 0.5) * width,
            accuracies,
            width,
            yerr=deviations,
            label=classifier_name,
            capsize=3,
        )
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (50%)")
    axis.set(
        xlabel="Subject",
        ylabel="Accuracy",
        title="Within-Subject Music Preference Classification",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels([f"s{subject:02d}" for subject in subjects], rotation=45)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper right")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, bbox_inches="tight")
    return figure


def plot_cross_subject_accuracy(
    results: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> plt.Figure:
    setup_style()
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in names]
    deviations = [results[name]["accuracy_std"] for name in names]

    figure, axis = plt.subplots(figsize=(8, 6))
    bars = axis.bar(
        names, accuracies, yerr=deviations, capsize=5, color=sns.color_palette()
    )
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (50%)")
    axis.set(
        ylabel="Accuracy",
        title="Cross-Subject Music Preference Classification\n(Leave-One-Subject-Out)",
    )
    axis.set_ylim(0, 1)
    axis.legend()
    for bar, accuracy, deviation in zip(bars, accuracies, deviations):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + deviation + 0.02,
            f"{accuracy:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, bbox_inches="tight")
    return figure


def plot_temporal_decoding(
    temporal_results: dict[str, np.ndarray],
    save_path: Path | None = None,
) -> plt.Figure:
    setup_style()
    times = temporal_results["center_times"]
    mean_accuracy = temporal_results["mean_accuracy"]
    std_accuracy = temporal_results["std_accuracy"]

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(times, mean_accuracy, color="steelblue", linewidth=2, label="Mean accuracy")
    axis.fill_between(
        times,
        mean_accuracy - std_accuracy,
        mean_accuracy + std_accuracy,
        alpha=0.3,
        color="steelblue",
    )
    if "subject_accuracies" in temporal_results:
        for subject_accuracy in temporal_results["subject_accuracies"]:
            axis.plot(times, subject_accuracy, alpha=0.2, color="gray", linewidth=0.5)
    axis.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance (50%)")
    axis.set(
        xlabel="Time from stimulus onset (seconds)",
        ylabel="Classification Accuracy",
        title="Time-Resolved Music Preference Decoding",
    )
    axis.set_ylim(0.3, 0.9)
    axis.legend()
    axis.grid(True, alpha=0.3)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, bbox_inches="tight")
    return figure


def plot_shap_importance(
    shap_results: dict,
    top_k: int = 20,
    save_path: Path | None = None,
) -> plt.Figure:
    setup_style()
    mean_abs = shap_results["mean_abs_shap"]
    names = shap_results["feature_names"]
    sorted_indices = np.argsort(mean_abs)[::-1][:top_k]

    figure, axis = plt.subplots(figsize=(10, 8))
    y_positions = np.arange(top_k)
    axis.barh(y_positions, mean_abs[sorted_indices][::-1], color="steelblue")
    axis.set_yticks(y_positions)
    axis.set_yticklabels([names[index] for index in sorted_indices][::-1])
    axis.set(
        xlabel="Mean |SHAP value|", title=f"Top {top_k} Most Important Features (SHAP)"
    )
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, bbox_inches="tight")
    return figure


def plot_band_importance(
    shap_results: dict,
    save_path: Path | None = None,
) -> plt.Figure:
    setup_style()
    band_names = shap_results["band_names"]
    band_importance = shap_results["band_importance"]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        band_names, band_importance, color=sns.color_palette("viridis", len(band_names))
    )
    axis.set(
        xlabel="Frequency Band",
        ylabel="Aggregated |SHAP| Importance",
        title="Feature Importance by Frequency Band",
    )
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, bbox_inches="tight")
    return figure


def plot_channel_topomap(
    channel_importance: np.ndarray,
    channel_names: list[str] | None = None,
    title: str = "Channel Importance (SHAP)",
    save_path: Path | None = None,
) -> plt.Figure:
    setup_style()
    if channel_names is None:
        channel_names = EEG_CHANNELS
    info = mne.create_info(ch_names=channel_names, sfreq=128, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("biosemi32"))

    figure, axis = plt.subplots(figsize=(8, 8))
    mne.viz.plot_topomap(
        channel_importance, info, axes=axis, show=False, cmap="RdBu_r", contours=0
    )
    axis.set_title(title)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, bbox_inches="tight")
    return figure


def plot_eegnet_comparison(
    classical_results: dict[int, dict[str, dict[str, float]]],
    eegnet_results: dict[int, dict[str, float]],
    save_path: Path | None = None,
) -> plt.Figure:
    setup_style()
    common_subjects = sorted(set(classical_results) & set(eegnet_results))
    count = len(common_subjects)
    best_classical = [
        max(
            classical_results[subject][name]["accuracy"]
            for name in classical_results[subject]
        )
        for subject in common_subjects
    ]
    eegnet_accuracies = [
        eegnet_results[subject]["accuracy"] for subject in common_subjects
    ]

    figure, axis = plt.subplots(figsize=(max(10, count * 0.8), 6))
    x_positions = np.arange(count)
    bar_width = 0.35
    axis.bar(
        x_positions - bar_width / 2,
        best_classical,
        bar_width,
        label="Best Classical ML",
        color="steelblue",
    )
    axis.bar(
        x_positions + bar_width / 2, eegnet_accuracies, bar_width, label="EEGNet", color="coral"
    )
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (50%)")
    axis.set(
        xlabel="Subject",
        ylabel="Accuracy",
        title="Classical ML vs EEGNet: Within-Subject Comparison",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels([f"s{subject:02d}" for subject in common_subjects], rotation=45)
    axis.set_ylim(0, 1)
    axis.legend()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, bbox_inches="tight")
    return figure
