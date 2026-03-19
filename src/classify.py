import numpy as np
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_classifiers() -> dict[str, Pipeline]:
    return {
        "SVM": Pipeline(
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
        ),
        "XGBoost": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        eval_metric="logloss",
                        random_state=42,
                        verbosity=0,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=None,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _select_variance(
    X_train: np.ndarray,
    X_test: np.ndarray,
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.var(X_train, axis=0) > threshold
    return X_train[:, keep], X_test[:, keep], np.where(keep)[0]


def _select_mutual_info(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = min(k, X_train.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    return (
        selector.fit_transform(X_train, y_train),
        selector.transform(X_test),
        selector.get_support(indices=True),
    )


def within_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    classifiers: dict[str, Pipeline] | None = None,
) -> dict[str, dict[str, float]]:
    if classifiers is None:
        classifiers = get_classifiers()

    results: dict[str, dict[str, float]] = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for name, model in classifiers.items():
        accuracies, f1_scores, auc_scores = [], [], []
        for train_index, test_index in skf.split(X, y):
            if len(np.unique(y[train_index])) < 2 or len(np.unique(y[test_index])) < 2:
                continue
            model.fit(X[train_index], y[train_index])
            prediction = model.predict(X[test_index])
            probabilities = model.predict_proba(X[test_index])[:, 1]
            accuracies.append(accuracy_score(y[test_index], prediction))
            f1_scores.append(f1_score(y[test_index], prediction))
            auc_scores.append(roc_auc_score(y[test_index], probabilities))

        results[name] = {
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "auc": float(np.mean(auc_scores)) if auc_scores else 0.0,
            "accuracy_std": float(np.std(accuracies)) if accuracies else 0.0,
            "f1_std": float(np.std(f1_scores)) if f1_scores else 0.0,
            "auc_std": float(np.std(auc_scores)) if auc_scores else 0.0,
        }

    return results


def majority_vote_cv(
    X: np.ndarray,
    y: np.ndarray,
    k_features: int = 50,
    n_folds: int = 5,
) -> dict[str, float]:
    min_class = int(np.unique(y, return_counts=True)[1].min())
    actual_folds = min(n_folds, min_class)
    if actual_folds < 2:
        return {"accuracy": 0.5, "f1": 0.5, "accuracy_std": 0.0, "f1_std": 0.0}

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    vote_accuracies: list[float] = []
    vote_f1_scores: list[float] = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        X_train, X_test, _ = _select_variance(X_train, X_test, threshold=1e-6)
        if X_train.shape[1] > k_features:
            X_train, X_test, _ = _select_mutual_info(
                X_train,
                y_train,
                X_test,
                k=k_features,
            )

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(X_train)
        test_scaled = scaler.transform(X_test)

        predictions = []
        for model in [
            XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            ),
            RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
            LGBMClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            ),
        ]:
            try:
                model.fit(train_scaled, y_train)
                predictions.append(model.predict(test_scaled))
            except Exception:
                continue

        if len(predictions) < 2:
            continue

        y_vote = stats.mode(np.array(predictions), axis=0, keepdims=False).mode
        vote_accuracies.append(float(accuracy_score(y_test, y_vote)))
        vote_f1_scores.append(float(f1_score(y_test, y_vote)))

    return {
        "accuracy": float(np.mean(vote_accuracies)) if vote_accuracies else 0.5,
        "f1": float(np.mean(vote_f1_scores)) if vote_f1_scores else 0.5,
        "accuracy_std": float(np.std(vote_accuracies)) if vote_accuracies else 0.0,
        "f1_std": float(np.std(vote_f1_scores)) if vote_f1_scores else 0.0,
    }
