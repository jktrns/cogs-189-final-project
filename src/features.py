import numpy as np
from scipy.signal import welch

from .data import EEG_CHANNELS, SAMPLING_RATE


FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

FRONTAL_PAIRS: list[tuple[str, str]] = [
    ("Fp2", "Fp1"),
    ("AF4", "AF3"),
    ("F4", "F3"),
    ("F8", "F7"),
]

_N_BANDS = len(FREQ_BANDS)
_N_CHANNELS = len(EEG_CHANNELS)


def compute_band_power(
    signal: np.ndarray,
    band: tuple[float, float] = (8.0, 13.0),
    fs: int = SAMPLING_RATE,
    nperseg: int = 256,
) -> float:
    nperseg = min(nperseg, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.mean(psd[mask])) if np.any(mask) else 0.0


def compute_differential_entropy(
    signal: np.ndarray,
    band: tuple[float, float] = (8.0, 13.0),
    fs: int = SAMPLING_RATE,
    nperseg: int = 256,
) -> float:
    power = compute_band_power(signal, band, fs, nperseg)
    return 0.5 * np.log(2 * np.pi * np.e * power) if power > 0 else -10.0


def compute_band_power_ratios(
    signal: np.ndarray,
    fs: int = SAMPLING_RATE,
    nperseg: int = 256,
) -> np.ndarray:
    nperseg = min(nperseg, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    band_powers = {}
    for name, (low, high) in FREQ_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        band_powers[name] = max(
            float(np.mean(psd[mask])) if np.any(mask) else 1e-10, 1e-10
        )

    return np.array(
        [
            band_powers["alpha"] / band_powers["theta"],
            band_powers["beta"] / band_powers["alpha"],
            band_powers["theta"] / band_powers["beta"],
            band_powers["gamma"] / band_powers["beta"],
        ]
    )


def compute_petrosian_fd(signal: np.ndarray) -> float:
    length = len(signal)
    if length < 3:
        return 0.0
    sign_changes = int(np.sum(np.diff(np.sign(np.diff(signal))) != 0))
    if sign_changes == 0:
        return np.log10(length)
    return float(
        np.log10(length)
        / (np.log10(length) + np.log10(length / (length + 0.4 * sign_changes)))
    )


def compute_spectral_entropy(
    signal: np.ndarray,
    fs: int = SAMPLING_RATE,
    nperseg: int = 256,
) -> float:
    nperseg = min(nperseg, len(signal))
    _, psd = welch(signal, fs=fs, nperseg=nperseg)
    distribution = psd / (np.sum(psd) + 1e-10)
    distribution = distribution[distribution > 0]
    return float(-np.sum(distribution * np.log2(distribution)))


def extract_differential_entropy(
    trial: np.ndarray,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    n_channels = trial.shape[0]
    features = np.zeros(n_channels * _N_BANDS)
    for channel in range(n_channels):
        for band_index, band_range in enumerate(FREQ_BANDS.values()):
            features[channel * _N_BANDS + band_index] = compute_differential_entropy(
                trial[channel], band_range, fs
            )
    return features


def extract_frontal_alpha_asymmetry(
    trial: np.ndarray,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    alpha = FREQ_BANDS["alpha"]
    result = np.zeros(len(FRONTAL_PAIRS))
    for pair_index, (right_channel, left_channel) in enumerate(FRONTAL_PAIRS):
        right_power = compute_band_power(
            trial[EEG_CHANNELS.index(right_channel)], alpha, fs
        )
        left_power = compute_band_power(
            trial[EEG_CHANNELS.index(left_channel)], alpha, fs
        )
        right_log = np.log(right_power) if right_power > 0 else -10.0
        left_log = np.log(left_power) if left_power > 0 else -10.0
        result[pair_index] = right_log - left_log
    return result


def extract_all_features(
    trial: np.ndarray,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    return np.concatenate(
        [
            extract_differential_entropy(trial, fs),
            extract_frontal_alpha_asymmetry(trial, fs),
        ]
    )


def _extract_for_subject(trial_data: np.ndarray, extractor, fs: int) -> np.ndarray:
    first = extractor(trial_data[0], fs)
    output = np.zeros((trial_data.shape[0], len(first)))
    output[0] = first
    for trial_index in range(1, len(trial_data)):
        output[trial_index] = extractor(trial_data[trial_index], fs)
    return output


def extract_features_for_subject(
    data: np.ndarray,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    return _extract_for_subject(data, extract_all_features, fs)


def get_feature_names() -> list[str]:
    names = [f"DE_{channel}_{band}" for channel in EEG_CHANNELS for band in FREQ_BANDS]
    names += [f"FAA_{right}-{left}" for right, left in FRONTAL_PAIRS]
    return names
