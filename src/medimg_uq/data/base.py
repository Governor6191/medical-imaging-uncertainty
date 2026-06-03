"""The base medical dataset.

A thin abstract class that pins down the contract: a dataset knows its task and
how many classes it has, and every item it returns is a ``Sample``. ISIC and
BraTS adapters subclass this. Nothing here is task-specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from medimg_uq.contract import Sample, Task


class MedicalDataset(ABC, Dataset):
    """Common base for every dataset adapter.

    Subclasses set ``task`` and ``num_classes`` and implement ``__len__`` and
    ``__getitem__``. ``__getitem__`` must return a ``Sample`` whose ``task``
    matches ``self.task``.

    Args:
        num_classes: Number of target classes. For binary classification this is
            2 (we model both classes rather than a single logit, so the same
            calibration code serves binary and multi-class without a special case).
    """

    task: Task

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.num_classes = num_classes

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Sample: ...

    def describe(self) -> str:
        """One-line human-readable summary, handy in logs and reports."""
        return (
            f"{type(self).__name__}(task={self.task}, "
            f"num_classes={self.num_classes}, n={len(self)})"
        )
