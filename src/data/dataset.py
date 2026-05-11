"""
CTSurvivalDataset — loads preprocessed tensors and clinical labels.
"""

import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class CTSurvivalDataset(Dataset):
    """
    Dataset for paired CT volumes and clinical tabular features
    with censored survival labels.

    Args:
        labels_csv  : path to CSV with columns:
                      patient_id, survival_days, event, age, stage, gender
        tensor_dir  : directory containing patient_NNN.pt files
                      (shape: [1, D, H, W], values in [0, 1])

    Returns per item:
        ct_tensor   : FloatTensor [1, 64, 64, 64]
        tabular     : FloatTensor [3]  (age/100, stage/4, gender)
        time        : FloatTensor scalar
        event       : FloatTensor scalar  (1=event, 0=censored)
    """

    TABULAR_FEATURES = ["age", "stage", "gender"]
    TABULAR_SCALE    = [100.0, 4.0, 1.0]

    def __init__(self, labels_csv: str | Path, tensor_dir: str | Path):
        self.df         = pd.read_csv(labels_csv)
        self.tensor_dir = Path(tensor_dir)
        self._validate()

    def _validate(self):
        required = {"patient_id", "survival_days", "event", "age", "stage", "gender"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"labels CSV missing columns: {missing}")
        expected_tensors = [
            self.tensor_dir / f"{pid}.pt" for pid in self.df["patient_id"]
        ]
        missing_tensors = [p for p in expected_tensors if not p.exists()]
        if missing_tensors:
            raise FileNotFoundError(
                f"{len(missing_tensors)} tensor files not found. "
                f"First missing: {missing_tensors[0]}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row       = self.df.iloc[idx]
        ct_tensor = torch.load(
            self.tensor_dir / f"{row['patient_id']}.pt",
            weights_only=True
        ).float()

        tabular = torch.tensor(
            [row[f] / s for f, s in zip(self.TABULAR_FEATURES, self.TABULAR_SCALE)],
            dtype=torch.float32,
        )

        time  = torch.tensor(float(row["survival_days"]), dtype=torch.float32)
        event = torch.tensor(float(row["event"]),         dtype=torch.float32)

        return ct_tensor, tabular, time, event

    def get_dataframe(self) -> pd.DataFrame:
        """return the underlying labels dataframe."""
        return self.df.copy()
