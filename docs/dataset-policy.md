# Dataset policy

Every dataset is described by a strict `DatasetManifest` with identity/version, task/domain, license/status/reference, source/citation, allowed use, redistribution terms, provenance, deterministic splits, sample hashes, annotation types, transforms, hashes, and warnings. Every sample includes identity/path/SHA-256/dimensions/channels/pixel format/domain/split/annotations/source/license reference.

License statuses are deliberately explicit:

- `redistributable` and `synthetic` can support public work when all stated terms are met.
- `download_script_only` keeps source images out of Git.
- `local_private` remains outside Git and cannot feed production artifacts by default.
- `research_only`, `commercially_unsafe`, and `unknown` are blocked from production artifacts; `unknown` is blocked entirely.

Public benchmark metadata may be committed without restricted images. Synthetic fixtures are generated deterministically; private and download-only data must preserve their manifest, citations, sample hashes, transform record, and reproducible split seed without redistribution.

```powershell
python -m wellfriend_models.datasets.validate path/to/dataset_manifest.json
python -m wellfriend_models.datasets.validate path/to/dataset_manifest.json --production-use
```
