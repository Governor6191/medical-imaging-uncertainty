"""medimg_uq: a modality-agnostic medical-imaging framework with calibrated uncertainty.

The package is split into a shared core and thin task modules. The core
(``uq``, ``calibration``, ``train``, ``eval``) is modality-agnostic and never
imports anything ISIC-specific or BraTS-specific. Task differences live behind
the ``data`` adapter contract and the ``models`` factory.
"""

from medimg_uq.contract import Sample, Task

__version__ = "0.1.0"

__all__ = ["Sample", "Task", "__version__"]
