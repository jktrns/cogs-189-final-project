import pickle
from pathlib import Path

import numpy as np


SAMPLING_RATE = 128
N_SUBJECTS = 32
N_TRIALS = 40
N_EEG_CHANNELS = 32

AVAILABLE_SUBJECTS = list(range(1, N_SUBJECTS + 1))

BASELINE_DURATION = 3
TRIAL_DURATION = 60
BASELINE_SAMPLES = BASELINE_DURATION * SAMPLING_RATE
TRIAL_SAMPLES = TRIAL_DURATION * SAMPLING_RATE

EEG_CHANNELS = [
    "Fp1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
    "Oz",
    "Pz",
    "Fp2",
    "AF4",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "C4",
    "T8",
    "CP6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
    "Fz",
    "Cz",
]

LABEL_VALENCE = 0
LABEL_AROUSAL = 1
LABEL_DOMINANCE = 2
LABEL_LIKING = 3

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "deap"


def load_subject(
    subject_id: int,
    data_dir: Path = DEFAULT_DATA_DIR,
    eeg_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= subject_id <= N_SUBJECTS:
        raise ValueError(f"subject_id must be 1–{N_SUBJECTS}, got {subject_id}")

    path = data_dir / f"s{subject_id:02d}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"Not found: {path}\n" f"Download DEAP and extract .dat files to {data_dir}"
        )

    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    data = raw["data"][:, :N_EEG_CHANNELS, :] if eeg_only else raw["data"]
    return data, raw["labels"]


def load_all_subjects(
    data_dir: Path = DEFAULT_DATA_DIR,
    eeg_only: bool = True,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        subject: load_subject(subject, data_dir, eeg_only)
        for subject in AVAILABLE_SUBJECTS
        if (data_dir / f"s{subject:02d}.dat").exists()
    }


def get_trial_data(data: np.ndarray, include_baseline: bool = False) -> np.ndarray:
    return data if include_baseline else data[:, :, BASELINE_SAMPLES:]


def binarize_liking(
    labels: np.ndarray,
    method: str = "median",
    threshold: float = 5.0,
) -> np.ndarray:
    liking = labels[:, LABEL_LIKING]
    cutoff = np.median(liking) if method == "median" else threshold
    return (liking > cutoff).astype(int)


def binarize_with_margin(
    labels: np.ndarray,
    margin: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    liking = labels[:, LABEL_LIKING]
    median = np.median(liking)
    keep = np.abs(liking - median) > margin
    return (liking[keep] > median).astype(int), keep


def binarize_quantile(
    labels: np.ndarray,
    lower_q: float = 0.33,
    upper_q: float = 0.67,
) -> tuple[np.ndarray, np.ndarray]:
    liking = labels[:, LABEL_LIKING]
    low, high = np.quantile(liking, lower_q), np.quantile(liking, upper_q)
    keep = (liking <= low) | (liking >= high)
    return (liking[keep] >= high).astype(int), keep


def get_best_binarization(
    labels: np.ndarray,
    min_trials: int = 20,
) -> tuple[np.ndarray, np.ndarray, str]:
    candidates = [
        ("margin_0.5", *binarize_with_margin(labels, 0.5)),
        ("margin_1.0", *binarize_with_margin(labels, 1.0)),
        ("quantile_33_67", *binarize_quantile(labels)),
    ]
    candidates = [
        (name, binary, mask)
        for name, binary, mask in candidates
        if len(binary) >= min_trials
    ]

    if not candidates:
        binary = binarize_liking(labels)
        return binary, np.ones(len(labels), dtype=bool), "median"

    def score_binarization(binary: np.ndarray) -> float:
        balance = min(binary.mean(), 1 - binary.mean()) * 2
        return 0.6 * balance + 0.4 * len(binary) / len(labels)

    best = max(candidates, key=lambda candidate: score_binarization(candidate[1]))
    return best[1], best[2], best[0]
