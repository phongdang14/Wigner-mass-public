"""
Model evaluation utilities for symass.

Public API
----------
evaluate -- compute RMSE, MAE, y_true, y_pred for any model type

The function is deliberately model-agnostic: the caller provides a ``pred_fn``
that converts ``(model, X)`` into a plain numpy prediction array, so the same
function works for a plain NN, an MDN, an XGBoost BDT, or any future model.

Examples
--------
Plain NN (default — no pred_fn needed)::

    from symass.validate import evaluate
    metrics = evaluate(nn_model, X_te, y_te, norm=norm)

MDN::

    pred_fn = lambda m, X: mdn_predict(*m(X)).detach().numpy()
    metrics = evaluate(mdn_model, X_te, y_te, norm=norm, pred_fn=pred_fn)

XGBoost BDT (raw numpy, no normaliser)::

    pred_fn = lambda m, X: (
        np.stack([mm.predict(X) for mm in m], axis=1)
        if isinstance(m, list) else m.predict(X).reshape(-1, 1)
    )
    metrics = evaluate(bdt_model, X_te, y_te, pred_fn=pred_fn)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def _default_pred_fn(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Plain torch forward pass under no_grad; output is the prediction."""
    model.eval()
    with torch.no_grad():
        return model(X).numpy()


def evaluate(
    model,
    X,
    y,
    norm=None,
    pred_fn: Callable | None = None,
) -> dict:
    """
    Evaluate model predictions against ground truth.

    Parameters
    ----------
    model   : Trained model — PyTorch ``nn.Module`` or any sklearn-style
              regressor (e.g. XGBoost).
    X       : Input features.  ``torch.Tensor`` for NN / MDN;
              ``np.ndarray`` for BDT.
    y       : Ground-truth targets, in normalised space when ``norm`` is given,
              otherwise in raw units.  Shape ``(n,)`` or ``(n, T)``.
    norm    : :class:`symass.Normalizer` fitted on the training set.
              When provided both ``y_pred`` and ``y_true`` are
              inverse-transformed before computing metrics.  Pass ``None``
              (default) when predictions are already in the original unit
              (e.g. BDT trained on raw targets).
    pred_fn : ``(model, X) -> np.ndarray`` of shape ``(n, T)``.

              Converts model output to a point-prediction array.  The default
              (``None``) calls ``model(X).numpy()`` under ``torch.no_grad()``,
              which is correct for a plain NN.

              MDN example::

                  pred_fn = lambda m, X: mdn_predict(*m(X)).detach().numpy()

              BDT example::

                  pred_fn = lambda m, X: (
                      np.stack([mm.predict(X) for mm in m], axis=1)
                      if isinstance(m, list) else m.predict(X).reshape(-1, 1)
                  )

    Returns
    -------
    dict
        ``"rmse"``   — float, root-mean-square error in target units.
        ``"mae"``    — float, mean absolute error in target units.
        ``"y_true"`` — ``np.ndarray`` shape ``(n,)``, ground truth in target units.
        ``"y_pred"`` — ``np.ndarray`` shape ``(n,)``, predictions in target units.
    """
    if pred_fn is None:
        pred_fn = _default_pred_fn

    y_pred_raw = pred_fn(model, X)
    y_true_raw = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

    if norm is not None:
        y_pred = norm.inverse_y(y_pred_raw)
        y_true = norm.inverse_y(y_true_raw)
    else:
        y_pred = y_pred_raw
        y_true = y_true_raw

    residuals = y_pred.flatten() - y_true.flatten()
    return {
        "rmse":   float(np.sqrt(np.mean(residuals ** 2))),
        "mae":    float(np.mean(np.abs(residuals))),
        "y_true": y_true.flatten(),
        "y_pred": y_pred.flatten(),
    }
