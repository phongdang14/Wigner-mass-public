"""
Generic training loops for symass models.

Public API
----------
build_gk_indices  -- precompute Garvey-Kelson 6-body index table
train             -- single epoch, data loss only
train_pinn        -- single epoch, data loss + Garvey-Kelson physics term

Both ``train`` and ``train_pinn`` are model-agnostic: the caller supplies a
``loss_fn`` that maps (model_output, y_batch) → scalar tensor, so they work
unchanged for a plain NN (MSE), an MDN (NLL), or any other architecture.

Example
-------
Plain NN::

    from symass.training import build_gk_indices, train, train_pinn
    import torch.nn as nn

    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        loss = train_pinn(nn_model, loader, opt, loss_fn,
                          X_full, gk_idx, gk_lambda=0.1)

MDN::

    from symass.training import train, train_pinn

    loss_fn = lambda out, y: mdn_loss(*out, y)
    for epoch in range(epochs):
        loss = train_pinn(mdn_model, loader, opt, loss_fn,
                          X_full, gk_idx,
                          pred_fn=mdn_predict, gk_lambda=0.1)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Garvey-Kelson index builder ───────────────────────────────────────────────

def build_gk_indices(N_arr: np.ndarray, Z_arr: np.ndarray) -> torch.Tensor | None:
    """
    Precompute the 6-body index tuples for the Garvey-Kelson mass relations.

    For each nucleus i, we look for the five neighbours that complete one of
    the two GK equations.  When all six nuclei are present in the dataset the
    six sample indices are recorded.

    GK Eq.(1)  [neutron-rich,  N ≥ Z]:
        BE(Z-2,N+2) - BE(Z,N) + BE(Z-1,N) - BE(Z-2,N+1) + BE(Z,N+1) - BE(Z-1,N+2) ≈ 0

    GK Eq.(2)  [proton-rich,   Z > N]:
        BE(Z+2,N-2) - BE(Z,N) + BE(Z,N-1) - BE(Z+1,N-2) + BE(Z+1,N) - BE(Z+2,N-1) ≈ 0

    The sign pattern for the residual is always  +1 -1 +1 -1 +1 -1  applied to
    the six indices stored in each row.

    Parameters
    ----------
    N_arr, Z_arr : Integer arrays of neutron / proton numbers for the dataset
                   (training split only — do not mix with test nuclei).

    Returns
    -------
    torch.Tensor of shape (M, 6) and dtype long, or None if no complete tuple
    is found.
    """
    zn_idx = {(int(z), int(n)): i
              for i, (z, n) in enumerate(zip(Z_arr, N_arr))}
    rows: list[list[int]] = []

    for z, n in zip(Z_arr, N_arr):
        z, n = int(z), int(n)
        if n >= z:
            keys = [(z-2,n+2), (z,n), (z-1,n), (z-2,n+1), (z,n+1), (z-1,n+2)]
        else:
            keys = [(z+2,n-2), (z,n), (z,n-1), (z+1,n-2), (z+1,n), (z+2,n-1)]
        if all(k in zn_idx for k in keys):
            rows.append([zn_idx[k] for k in keys])

    if not rows:
        return None
    return torch.tensor(rows, dtype=torch.long)


def _gk_physics_loss(y_pred: torch.Tensor, gk_idx: torch.Tensor) -> torch.Tensor:
    """
    Mean squared Garvey-Kelson residual.

    y_pred : (N, T)  — full-dataset point predictions (all training nuclei).
    gk_idx : (M, 6)  — index table from build_gk_indices().
    """
    m = y_pred[gk_idx]                               # (M, 6, T)
    residual = m[:,0] - m[:,1] + m[:,2] - m[:,3] + m[:,4] - m[:,5]
    return residual.pow(2).mean()


# ── Training loops ─────────────────────────────────────────────────────────────

def _maybe_print(label: str, epoch: int, loss: float, print_every: int) -> None:
    """Print a one-line loss summary when the epoch hits the reporting interval."""
    if print_every > 0 and epoch % print_every == 0:
        print(f"  [{label}]  epoch {epoch:>6d}  loss {loss:.6f}")


def _eval_data_loss(model: nn.Module, loss_fn: Callable, test_data: tuple) -> float:
    """
    Data loss on a held-out set, in the same space as the training data loss.

    ``test_data`` is an ``(X_test, y_test)`` pair of tensors.  Evaluated under
    ``model.eval()`` / ``no_grad`` so dropout/batch-norm and gradients are off;
    the caller's training loop restores ``model.train()`` at the next epoch.
    The returned value is directly comparable to the per-epoch training loss,
    making it suitable for spotting overfitting / choosing an early-stopping point.
    """
    X_test, y_test = test_data
    model.eval()
    with torch.no_grad():
        loss = loss_fn(model(X_test), y_test)
    return float(loss.item())


class _EarlyStopper:
    """
    Early stopping on the held-out (test) loss.

    Call :meth:`step` once per epoch with the current test loss.  It tracks the
    best loss seen, snapshots the model weights at that point, and returns
    ``True`` once ``patience`` epochs have passed with no improvement larger
    than ``min_delta``.  After the loop, :meth:`restore` reloads the best
    weights so the returned model is the best one, not the last.
    """

    def __init__(self, model: nn.Module, patience: int, min_delta: float,
                 restore_best: bool):
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self._bad = 0

    def step(self, test_loss: float, epoch: int) -> bool:
        """Update state; return ``True`` if training should stop now."""
        if test_loss < self.best - self.min_delta:
            self.best = test_loss
            self.best_epoch = epoch
            self._bad = 0
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in self.model.state_dict().items()}
            return False
        self._bad += 1
        return self._bad >= self.patience

    def restore(self) -> None:
        """Reload the best-epoch weights into the model (if tracked)."""
        if self.restore_best and self.best_state is not None:
            self.model.load_state_dict(self.best_state)


def train(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_fn: Callable,
    epochs: int = 100,
    print_every: int = 0,
    label: str = "train",
    grad_clip: float | None = None,
    test_data: tuple | None = None,
    early_stopping: bool = False,
    patience: int = 50,
    min_delta: float = 0.0,
    restore_best: bool = True,
) -> list[float] | tuple[list[float], list[float]]:
    """
    Full training loop — data loss only.

    Parameters
    ----------
    model       : PyTorch module in training mode (set internally each epoch).
    loader      : DataLoader yielding ``(X_batch, y_batch)`` pairs.
    optimiser   : Any PyTorch-compatible optimiser.
    loss_fn     : ``(model_output, y_batch) -> scalar tensor``.

                  Plain NN example::

                      loss_fn = torch.nn.MSELoss()

                  MDN example::

                      loss_fn = lambda out, y: mdn_loss(*out, y)

    epochs      : Number of training epochs (default 100).
    print_every : Print mean loss every this many epochs.  ``0`` disables
                  printing entirely.
    label       : Short tag shown in each printed line, e.g. ``"NN"`` or
                  ``"MDN"`` (default ``"train"``).
    grad_clip   : If set, clip gradient L2-norm to this value before each
                  optimiser step.  Recommended for MDN (e.g. ``1.0``).
    test_data   : Optional ``(X_test, y_test)`` pair for monitoring.  When
                  provided, the held-out data loss (evaluated under
                  ``model.eval()``) is recorded every epoch and reported on
                  each ``print_every`` line, letting you watch for overfitting.
                  ``None`` (default) leaves printing and the return value
                  unchanged.  Required when ``early_stopping=True``.
    early_stopping : If ``True``, stop once the held-out ``test_data`` loss has
                  not improved for ``patience`` epochs, and (by default) restore
                  the best-epoch weights.  Requires ``test_data`` (default
                  ``False``).
    patience    : Epochs of no test-loss improvement tolerated before stopping
                  (default 50).
    min_delta   : Minimum decrease in test loss counted as an improvement
                  (default 0.0).
    restore_best : If ``True`` (default), reload the lowest-test-loss weights
                  when training ends (whether by early stop or by exhausting
                  ``epochs``).

    Returns
    -------
    list[float] | tuple[list[float], list[float]]
        Without ``test_data``: the per-epoch mean training loss.
        With ``test_data``: a ``(train_history, test_history)`` pair, both
        one value per epoch and directly comparable — ready to plot as
        learning curves.  With early stopping the histories run only up to the
        epoch where training stopped.
    """
    if early_stopping and test_data is None:
        raise ValueError("early_stopping=True requires test_data=(X_test, y_test).")
    stopper = (_EarlyStopper(model, patience, min_delta, restore_best)
               if early_stopping else None)
    history: list[float] = []
    test_history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for X_batch, y_batch in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            total += loss.item() * len(X_batch)
        mean_loss = total / len(loader.dataset)
        history.append(mean_loss)

        test_loss = None
        if test_data is not None:
            test_loss = _eval_data_loss(model, loss_fn, test_data)
            test_history.append(test_loss)

        if print_every > 0 and epoch % print_every == 0:
            if test_loss is not None:
                print(f"  [{label}]  epoch {epoch:>6d}  "
                      f"train {mean_loss:.6f}  test {test_loss:.6f}")
            else:
                _maybe_print(label, epoch, mean_loss, print_every)

        if stopper is not None and stopper.step(test_loss, epoch):
            if print_every > 0:
                print(f"  [{label}]  early stop at epoch {epoch}  "
                      f"(best test {stopper.best:.6f} @ epoch {stopper.best_epoch})")
            break

    if stopper is not None:
        stopper.restore()
    return (history, test_history) if test_data is not None else history


def train_pinn(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_fn: Callable,
    X_full: torch.Tensor,
    gk_idx: torch.Tensor,
    pred_fn: Callable | None = None,
    gk_lambda: float = 1.0,
    adaptive_gk: bool = False,
    gk_ratio: float = 1.0,
    ema_momentum: float = 0.9,
    lambda_max: float = 100.0,
    epochs: int = 100,
    print_every: int = 0,
    label: str = "pinn",
    grad_clip: float | None = None,
    test_data: tuple | None = None,
    early_stopping: bool = False,
    patience: int = 50,
    min_delta: float = 0.0,
    restore_best: bool = True,
) -> list[float] | tuple[list[float], list[float]]:
    """
    Full training loop with data loss + Garvey-Kelson physics regularisation.

    Total loss per step::

        L = loss_fn(output, y_batch)  +  λ * GK_residual(y_full)

    where λ is either fixed (``adaptive_gk=False``) or updated each epoch by
    **loss-ratio balancing with EMA** (``adaptive_gk=True``).

    The GK residual is evaluated on the full training set every mini-batch step
    so the physics constraint is always applied to all available nuclear pairs.
    Only the data-loss component is tracked in the history so that training
    curves are directly comparable to plain ``train``.

    **Adaptive λ (loss-ratio balancing)**

    The fixed-λ approach has a well-known pathology: as the model improves,
    ``L_data`` shrinks while ``L_gk`` stays near zero, so the relative weight of
    the physics term balloons and can dominate training.  Adaptive balancing
    sets a *target ratio* between the two terms::

        λ_target = gk_ratio * L_data / L_gk

    so that ``λ · L_gk ≈ gk_ratio · L_data`` throughout training.  An
    exponential moving average (EMA) dampens oscillations::

        λ ← ema_momentum * λ + (1 - ema_momentum) * λ_target

    Parameters
    ----------
    model        : PyTorch module.
    loader       : DataLoader yielding ``(X_batch, y_batch)`` pairs.
    optimiser    : Any PyTorch-compatible optimiser.
    loss_fn      : ``(model_output, y_batch) -> scalar tensor``.
    X_full       : Full training feature tensor ``(n_train, n_features)``.
    gk_idx       : ``(M, 6)`` long tensor from :func:`build_gk_indices`.
    pred_fn      : Optional callable ``model_output -> (n, T)`` point-prediction
                   tensor.  Pass ``None`` for a plain NN (the output is already
                   the prediction).  Pass ``mdn_predict`` for an MDN.
    gk_lambda    : Initial (or fixed) weight of the GK physics term.  When
                   ``adaptive_gk=True`` this is the starting value for the EMA;
                   when ``False`` it is constant throughout (default 1.0).
    adaptive_gk  : If ``True``, update λ each epoch via loss-ratio balancing
                   (default ``False`` — backward-compatible).
    gk_ratio     : Target ratio ``λ · L_gk / L_data`` when ``adaptive_gk=True``
                   (default 1.0 — physics and data losses contribute equally).
    ema_momentum : EMA smoothing factor for the adaptive λ update (default 0.9).
                   Higher → slower adaptation, lower → faster but noisier.
    lambda_max   : Upper bound on λ when ``adaptive_gk=True`` (default 100.0).
                   Prevents runaway growth when ``L_gk`` is near-zero (the GK
                   relations are already well satisfied in normalised space).
                   Set to ``float("inf")`` to disable clamping.
    epochs       : Number of training epochs (default 100).
    print_every  : Print data loss (and current λ when adaptive) every this
                   many epochs.  ``0`` disables printing.
    label        : Short tag shown in each printed line (default ``"pinn"``).
    grad_clip    : If set, clip gradient L2-norm to this value before each
                   optimiser step.  Recommended for MDN (e.g. ``1.0``).
    test_data    : Optional ``(X_test, y_test)`` pair for monitoring.  When
                   provided, the held-out data loss (the ``loss_fn`` term only,
                   evaluated under ``model.eval()``) is recorded every epoch and
                   reported on each ``print_every`` line, directly comparable to
                   the training data loss for spotting overfitting.  ``None``
                   (default) leaves printing and the return value unchanged.
                   Required when ``early_stopping=True``.
    early_stopping : If ``True``, stop once the held-out ``test_data`` loss has
                   not improved for ``patience`` epochs, and (by default)
                   restore the best-epoch weights.  Requires ``test_data``
                   (default ``False``).  Early stopping keys on the data-loss
                   term only, not the GK residual.
    patience     : Epochs of no test-loss improvement tolerated before stopping
                   (default 50).
    min_delta    : Minimum decrease in test loss counted as an improvement
                   (default 0.0).
    restore_best : If ``True`` (default), reload the lowest-test-loss weights
                   when training ends.

    Returns
    -------
    list[float] | tuple[list[float], list[float]]
        Without ``test_data``: the per-epoch mean data loss (GK term excluded).
        With ``test_data``: a ``(train_history, test_history)`` pair, both the
        data-loss term only and directly comparable — ready to plot as
        learning curves.  With early stopping the histories run only up to the
        epoch where training stopped.
    """
    if early_stopping and test_data is None:
        raise ValueError("early_stopping=True requires test_data=(X_test, y_test).")
    stopper = (_EarlyStopper(model, patience, min_delta, restore_best)
               if early_stopping else None)
    history: list[float] = []
    test_history: list[float] = []
    _lambda = float(gk_lambda)   # running λ — updated each epoch if adaptive

    for epoch in range(1, epochs + 1):
        model.train()
        total_data = 0.0
        total_gk   = 0.0

        for X_batch, y_batch in loader:
            optimiser.zero_grad()

            # ── data loss ────────────────────────────────────────────────────
            out       = model(X_batch)
            data_loss = loss_fn(out, y_batch)

            # ── GK physics loss (full training set) ──────────────────────────
            out_full = model(X_full)
            if pred_fn is not None:
                y_full = pred_fn(*out_full) if isinstance(out_full, tuple) else pred_fn(out_full)
            else:
                y_full = out_full
            gk = _gk_physics_loss(y_full, gk_idx)

            (data_loss + _lambda * gk).backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            total_data += data_loss.item() * len(X_batch)
            total_gk   += gk.item()        * len(X_batch)

        mean_data = total_data / len(loader.dataset)
        mean_gk   = total_gk   / len(loader.dataset)
        history.append(mean_data)

        # ── NaN guard ────────────────────────────────────────────────────────
        if not np.isfinite(mean_data):
            print(f"  [{label}]  *** NaN/Inf loss at epoch {epoch} — stopping early ***")
            break

        test_loss = None
        if test_data is not None:
            test_loss = _eval_data_loss(model, loss_fn, test_data)
            test_history.append(test_loss)

        # ── Adaptive λ update (end of epoch, after all mini-batches) ─────────
        if adaptive_gk:
            # Clamp denominator so λ cannot explode when GK constraint is
            # already nearly satisfied (gk_loss ≈ 0 in normalised space).
            # Floor = 1% of data loss, so λ_target ≤ 100 × gk_ratio.
            denom = max(mean_gk, 0.01 * mean_data)
            lambda_target = gk_ratio * mean_data / denom
            _lambda = ema_momentum * _lambda + (1 - ema_momentum) * lambda_target
            _lambda = min(_lambda, lambda_max)     # hard ceiling

        if print_every > 0 and epoch % print_every == 0:
            if test_loss is not None:
                extra = (f"  λ_gk {_lambda:.4f}  gk_loss {mean_gk:.6f}"
                         if adaptive_gk else "")
                print(f"  [{label}]  epoch {epoch:>6d}  "
                      f"train {mean_data:.6f}  test {test_loss:.6f}{extra}")
            elif adaptive_gk:
                print(f"  [{label}]  epoch {epoch:>6d}  loss {mean_data:.6f}"
                      f"  λ_gk {_lambda:.4f}  gk_loss {mean_gk:.6f}")
            else:
                _maybe_print(label, epoch, mean_data, print_every)

        if stopper is not None and stopper.step(test_loss, epoch):
            if print_every > 0:
                print(f"  [{label}]  early stop at epoch {epoch}  "
                      f"(best test {stopper.best:.6f} @ epoch {stopper.best_epoch})")
            break

    if stopper is not None:
        stopper.restore()
    return (history, test_history) if test_data is not None else history
