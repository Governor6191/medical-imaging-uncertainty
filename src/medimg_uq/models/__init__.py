"""Model factory.

Builds a backbone plus the head the task needs: a classification head for ISIC,
a U-Net decoder for BraTS. The rest of the core sees one model interface.
"""

from medimg_uq.models.factory import ClassificationModel, build_model

__all__ = ["ClassificationModel", "build_model"]
