"""Calibration figures.

Reliability diagrams from a :class:`~medimg_uq.calibration.metrics.ReliabilityCurve`.
Drawn through the Agg canvas without ``pyplot`` so the code is headless-safe and
holds no global figure state. Per-voxel uncertainty-map overlays for segmentation
are added with the BraTS application.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from medimg_uq.calibration.metrics import ReliabilityCurve


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
