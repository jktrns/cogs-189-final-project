import numpy as np
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .data import SAMPLING_RATE
from .features import extract_all_features


def compute_shap_values(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_explain: np.ndarray,
    method: str = "xgboost",
) -> np.ndarray:
    if method == "xgboost":
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(X_train)
        explain_scaled = scaler.transform(X_explain)
        model.fit(train_scaled, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(explain_scaled)

    elif method == "svm":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        background = shap.kmeans(X_train, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_explain)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        raise ValueError(f"Unknown method: {method}")

    return np.array(shap_values)


def get_top_features(
    mean_abs_shap: np.ndarray,
    feature_names: list[str],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    indices = np.argsort(mean_abs_shap)[::-1][:top_k]
    return [(feature_names[index], float(mean_abs_shap[index])) for index in indices]


def time_resolved_classification(
    data: np.ndarray,
    y: np.ndarray,
    window_size: float = 5.0,
    step_size: float = 2.5,
    n_folds: int = 5,
    fs: int = SAMPLING_RATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_trials, n_channels, n_samples = data.shape
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)

    starts = list(range(0, n_samples - window_samples + 1, step_samples))
    n_windows = len(starts)
    center_times = np.array([(start + window_samples / 2) / fs for start in starts])
    window_accuracies = np.zeros((n_windows, n_folds))

    for window_index, start in enumerate(starts):
        trial_features = []
        for trial_index in range(n_trials):
            window_data = data[trial_index, :, start : start + window_samples]
            trial_features.append(extract_all_features(window_data, fs))
        X_window = np.array(trial_features)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
            ]
        )

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold_index, (train_index, test_index) in enumerate(skf.split(X_window, y)):
            if len(np.unique(y[train_index])) < 2:
                window_accuracies[window_index, fold_index] = 0.5
                continue
            model.fit(X_window[train_index], y[train_index])
            window_accuracies[window_index, fold_index] = model.score(
                X_window[test_index],
                y[test_index],
            )

    mean_accuracies = np.mean(window_accuracies, axis=1)
    std_accuracies = np.std(window_accuracies, axis=1)
    return center_times, mean_accuracies, std_accuracies


def find_decodable_time(
    center_times: np.ndarray,
    mean_accuracy: np.ndarray,
    threshold: float = 0.55,
) -> float | None:
    above = mean_accuracy >= threshold
    if not np.any(above):
        return None
    return float(center_times[int(np.argmax(above))])


def load_familiarity_ratings(
    ratings_path: str | None = None,
) -> dict[int, np.ndarray]:
    import pandas as pd
    from pathlib import Path

    if ratings_path is None:
        ratings_path = str(
            Path(__file__).parent.parent / "data" / "deap" / "participant_ratings.csv"
        )

    ratings_frame = pd.read_csv(ratings_path)
    ratings: dict[int, np.ndarray] = {}
    for participant in ratings_frame["Participant_id"].unique():
        participant_frame = ratings_frame[ratings_frame["Participant_id"] == participant].sort_values(
            "Experiment_id"
        )
        ratings[int(participant)] = participant_frame["Familiarity"].values.astype(float)
    return ratings


def split_by_familiarity(
    familiarity: np.ndarray,
    method: str = "median",
    threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "median":
        cutoff = np.median(familiarity)
    elif method == "threshold":
        cutoff = threshold
    else:
        raise ValueError(f"Unknown method: {method}")
    return familiarity > cutoff, familiarity <= cutoff


def run_familiarity_analysis(
    subjects: list[int] | None = None,
    n_folds: int = 5,
) -> dict[str, dict[str, float]]:
    from .data import AVAILABLE_SUBJECTS, binarize_liking, get_trial_data, load_subject
    from .features import extract_features_for_subject

    if subjects is None:
        subjects = AVAILABLE_SUBJECTS

    familiarity_ratings = load_familiarity_ratings()
    high_accuracies: list[float] = []
    low_accuracies: list[float] = []
    all_accuracies: list[float] = []

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
        ]
    )

    for subject in subjects:
        print(f"  Familiarity analysis for s{subject:02d}...", end=" ", flush=True)

        if subject not in familiarity_ratings:
            print("no familiarity data")
            continue

        data, labels = load_subject(subject, eeg_only=True)
        trial_data = get_trial_data(data, include_baseline=False)
        X = extract_features_for_subject(trial_data)
        y = binarize_liking(labels)
        familiarity = familiarity_ratings[subject]

        if len(np.unique(y)) < 2:
            print("skipped (single class)")
            continue

        total_accuracy = _cv_accuracy(model, X, y, n_folds)
        all_accuracies.append(total_accuracy)

        high_mask, low_mask = split_by_familiarity(familiarity)

        if np.sum(high_mask) >= n_folds and len(np.unique(y[high_mask])) >= 2:
            high_accuracies.append(
                _cv_accuracy(model, X[high_mask], y[high_mask], n_folds)
            )
        if np.sum(low_mask) >= n_folds and len(np.unique(y[low_mask])) >= 2:
            low_accuracies.append(
                _cv_accuracy(model, X[low_mask], y[low_mask], n_folds)
            )

        print(f"all={total_accuracy:.1%}")

    return {
        "all_trials": {
            "accuracy": float(np.mean(all_accuracies)) if all_accuracies else 0.0,
            "accuracy_std": float(np.std(all_accuracies)) if all_accuracies else 0.0,
            "n_subjects": len(all_accuracies),
        },
        "high_familiarity": {
            "accuracy": float(np.nanmean(high_accuracies)) if high_accuracies else 0.0,
            "accuracy_std": (
                float(np.nanstd(high_accuracies)) if high_accuracies else 0.0
            ),
            "n_subjects": len(high_accuracies),
        },
        "low_familiarity": {
            "accuracy": float(np.nanmean(low_accuracies)) if low_accuracies else 0.0,
            "accuracy_std": float(np.nanstd(low_accuracies)) if low_accuracies else 0.0,
            "n_subjects": len(low_accuracies),
        },
    }


def _cv_accuracy(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
) -> float:
    min_class_count = int(np.bincount(y.astype(int)).min())
    actual_folds = min(n_folds, len(y), min_class_count)
    if actual_folds < 2:
        return 0.5
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in skf.split(X, y):
        if len(np.unique(y[train_index])) < 2:
            continue
        model.fit(X[train_index], y[train_index])
        accuracies.append(model.score(X[test_index], y[test_index]))
    return float(np.mean(accuracies)) if accuracies else 0.5
