"""Calibration figures.

Reliability diagrams from a :class:`~medimg_uq.calibration.metrics.ReliabilityCurve`.
Drawn through the Agg canvas without ``pyplot`` so the code is headless-safe and
holds no global figure state. Per-voxel uncertainty-map overlays for segmentation
are added with the BraTS application.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from medimg_uq.calibration.metrics import ReliabilityCurve


def _to_numpy(array) -> np.ndarray:
    """Accept a torch tensor or an array-like and return a detached numpy array."""
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def reliability_diagram(
    curve: ReliabilityCurve,
    *,
    ece: float | None = None,
    title: str = "Reliability diagram",
) -> Figure:
    """Plot accuracy against confidence per bin, with the perfect-calibration diagonal.

    Bars below the diagonal are overconfident bins (accuracy lags confidence); bars
    above are underconfident. ``ece``, if given, is annotated on the plot.
    """
    edges = curve.bin_edges.detach().cpu()
    centers = ((edges[:-1] + edges[1:]) / 2).numpy()
    width = float((edges[1] - edges[0]).item())
    accuracy = curve.bin_accuracy.detach().cpu().numpy()
    count = curve.bin_count.detach().cpu().numpy()
    populated = count > 0

    fig = Figure(figsize=(5, 5), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect calibration")
    ax.bar(
        centers[populated],
        accuracy[populated],
        width=width * 0.9,
        color="#3b6ea5",
        edgecolor="white",
        label="accuracy",
    )
    # The gap between the bar top and the diagonal is the per-bin miscalibration.
    ax.bar(
        centers[populated],
        centers[populated] - accuracy[populated],
        bottom=accuracy[populated],
        width=width * 0.9,
        color="#d1495b",
        alpha=0.4,
        edgecolor="white",
        label="gap",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    if ece is not None:
        ax.text(
            0.97,
            0.05,
            f"ECE = {ece:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
        )
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save a figure to ``path`` (creating parent folders) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def segmentation_panels(
    image: object,
    prediction: object,
    uncertainty: object,
    target: object | None = None,
    *,
    title: str = "Segmentation with per-voxel uncertainty",
) -> Figure:
    """The headline segmentation figure: MRI, prediction, and uncertainty side by side.

    Args:
        image: One MRI modality slice ``(H, W)`` for the grayscale background.
        prediction: Predicted class mask ``(H, W)``, integers; 0 is background.
        uncertainty: Per-voxel uncertainty map ``(H, W)`` (epistemic entropy).
        target: Optional ground-truth mask ``(H, W)``; adds a panel when given.
        title: Figure title.

    The story the figure should tell: the uncertainty lights up along the tumor
    boundary, where the ensemble members disagree about where the edge is.
    """
    image = _to_numpy(image)
    prediction = _to_numpy(prediction)
    uncertainty = _to_numpy(uncertainty)

    panels: list[tuple[str, object]] = [("MRI", None), ("prediction", prediction)]
    if target is not None:
        panels.append(("ground truth", _to_numpy(target)))
    panels.append(("uncertainty", uncertainty))

    fig = Figure(figsize=(4 * len(panels), 4), dpi=120)
    FigureCanvasAgg(fig)
    for i, (label, data) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=10)
        if label == "uncertainty":
            heat = ax.imshow(data, cmap="magma")
            fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(image, cmap="gray")
            if data is not None:
                mask = np.ma.masked_where(np.asarray(data) == 0, data)
                ax.imshow(mask, cmap="autumn", alpha=0.5, vmin=1)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
