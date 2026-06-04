"""Data adapters.

Holds the common dataset contract plus the per-application loaders (ISIC, BraTS).
Everything a task needs to differ lives here, behind the shared ``Sample`` contract.
"""

from medimg_uq.data.base import MedicalDataset
from medimg_uq.data.brats import BraTSDataset, BraTSSliceDataset, center_fit
from medimg_uq.data.isic import ISICDataset, isic_transforms
from medimg_uq.data.synthetic import (
    SyntheticClassificationDataset,
    SyntheticSegmentationDataset,
)

__all__ = [
    "BraTSDataset",
    "BraTSSliceDataset",
    "ISICDataset",
    "MedicalDataset",
    "SyntheticClassificationDataset",
    "SyntheticSegmentationDataset",
    "center_fit",
    "isic_transforms",
]
