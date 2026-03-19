import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from .data import N_EEG_CHANNELS


class EEGNet(nn.Module):

    def __init__(
        self,
        n_channels: int = N_EEG_CHANNELS,
        n_samples: int = 7680,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
            ),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )
        self._flat_size = self._get_flat_size(n_channels, n_samples)
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self, n_channels: int, n_samples: int) -> int:
        x = torch.zeros(1, 1, n_channels, n_samples)
        return int(self.block2(self.block1(x)).numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block2(self.block1(x))
        return self.classifier(x.flatten(start_dim=1))


def train_eegnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> tuple[float, list[float], list[float]]:
    n_channels, n_samples = X_train.shape[1], X_train.shape[2]
    model = EEGNet(n_channels=n_channels, n_samples=n_samples).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    train_label_tensor = torch.LongTensor(y_train).to(device)
    test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    test_label_tensor = torch.LongTensor(y_test).to(device)

    loader = DataLoader(
        TensorDataset(train_tensor, train_label_tensor),
        batch_size=batch_size,
        shuffle=True,
    )
    train_losses: list[float] = []
    test_accuracies: list[float] = []

    for _ in range(n_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            test_accuracies.append(
                (model(test_tensor).argmax(1) == test_label_tensor)
                .float()
                .mean()
                .item()
            )

    return test_accuracies[-1], train_losses, test_accuracies


def run_eegnet_analysis(
    subjects: list[int] | None = None,
    n_folds: int = 5,
    n_epochs: int = 50,
) -> dict[int, dict[str, float]]:
    from .data import AVAILABLE_SUBJECTS, binarize_liking, get_trial_data, load_subject

    if subjects is None:
        subjects = AVAILABLE_SUBJECTS

    results: dict[int, dict[str, float]] = {}
    for subject in subjects:
        print(f"  EEGNet for s{subject:02d}...", end=" ", flush=True)
        data, labels = load_subject(subject, eeg_only=True)
        trial_data = get_trial_data(data, include_baseline=False)
        y = binarize_liking(labels)

        if len(np.unique(y)) < 2:
            print("skipped (single class)")
            continue

        X = trial_data.copy()
        for trial_index in range(X.shape[0]):
            mean = X[trial_index].mean(axis=1, keepdims=True)
            std = X[trial_index].std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            X[trial_index] = (X[trial_index] - mean) / std

        fold_accuracies: list[float] = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(X, y):
            if len(np.unique(y[train_index])) < 2:
                continue
            accuracy, _, _ = train_eegnet(
                X[train_index],
                y[train_index],
                X[test_index],
                y[test_index],
                n_epochs=n_epochs,
            )
            fold_accuracies.append(accuracy)

        mean_accuracy = float(np.mean(fold_accuracies)) if fold_accuracies else 0.0
        std_accuracy = float(np.std(fold_accuracies)) if fold_accuracies else 0.0
        results[subject] = {"accuracy": mean_accuracy, "accuracy_std": std_accuracy}
        print(f"{mean_accuracy:.1%}")

    return results
