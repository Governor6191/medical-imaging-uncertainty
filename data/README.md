# Data

Datasets are downloaded here and are not tracked in git (see `.gitignore`). Each
application documents its own download step and license.

```
data/
  isic/     ISIC dermoscopy images and labels (benign vs malignant)
  brats/    BraTS multi-modal MRI volumes (requires a data-use agreement)
```

## ISIC

Openly downloadable from the ISIC archive. The loader expects an `isic/` folder
with images and a labels CSV. See the ISIC data adapter for the exact layout and
the download helper.

## BraTS

Access requires registration and a data-use agreement through Synapse. Honor the
agreement's terms and cite the dataset. Do not commit any BraTS file to git.

Both datasets are public and de-identified, so no IRB is required. These are
research models, not cleared diagnostic tools.
