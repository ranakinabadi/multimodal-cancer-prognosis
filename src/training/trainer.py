"""
Training loop with checkpointing and early stopping.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional

from .losses import cox_partial_likelihood
from ..evaluation.metrics import concordance_index


class Trainer:
    """
    General-purpose trainer for survival models.

    All models must accept (ct, tabular) and return a (B,) risk score vector.
    This unified interface means the same trainer works for all four
    architectures (image-only, tabular-only, concat, attention).

    Args:
        model        : any model from src/models/
        train_loader : DataLoader yielding (ct, tab, time, event)
        val_loader   : DataLoader yielding (ct, tab, time, event)
        device       : torch device
        lr           : initial learning rate
        weight_decay : L2 regularisation
        epochs       : maximum training epochs
        patience     : ReduceLROnPlateau patience
        save_path    : if provided, saves best checkpoint here
    """

    def __init__(self,
                 model:        torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 device:       torch.device,
                 lr:           float = 1e-3,
                 weight_decay: float = 1e-4,
                 epochs:       int   = 50,
                 patience:     int   = 5,
                 save_path:    Optional[str | Path] = None):

        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs
        self.save_path    = Path(save_path) if save_path else None

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=patience
        )

        self.history = {"train_loss": [], "val_cindex": []}
        self.best_ci = 0.0

   
    #  internal helpers
    

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for ct, tab, time, event in self.train_loader:
            ct, tab     = ct.to(self.device),   tab.to(self.device)
            time, event = time.to(self.device), event.to(self.device)

            self.optimizer.zero_grad()
            risk = self.model(ct, tab)
            loss = cox_partial_likelihood(risk, time, event)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        all_risk, all_time, all_event = [], [], []

        for ct, tab, time, event in self.val_loader:
            ct, tab = ct.to(self.device), tab.to(self.device)
            all_risk.append(self.model(ct, tab).cpu())
            all_time.append(time)
            all_event.append(event)

        return concordance_index(
            torch.cat(all_risk),
            torch.cat(all_time),
            torch.cat(all_event),
        )

    
    #  public API
    

    def fit(self) -> dict:
        """Run training. Returns history dict."""
        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_ci     = self._validate()

            self.scheduler.step(val_ci)
            self.history["train_loss"].append(train_loss)
            self.history["val_cindex"].append(val_ci)

            if val_ci > self.best_ci:
                self.best_ci = val_ci
                if self.save_path:
                    self.save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), self.save_path)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val C-index: {val_ci:.4f} | "
                      f"Best: {self.best_ci:.4f}")

        # Restore best weights
        if self.save_path and self.save_path.exists():
            self.model.load_state_dict(torch.load(self.save_path,
                                                   weights_only=True))

        return self.history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """evaluate on any DataLoader. Returns C-index."""
        self.model.eval()
        all_risk, all_time, all_event = [], [], []

        for ct, tab, time, event in loader:
            ct, tab = ct.to(self.device), tab.to(self.device)
            all_risk.append(self.model(ct, tab).cpu())
            all_time.append(time)
            all_event.append(event)

        return concordance_index(
            torch.cat(all_risk),
            torch.cat(all_time),
            torch.cat(all_event),
        )
