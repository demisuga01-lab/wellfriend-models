# Dataset policy

`DatasetManifest` records dataset version, license, samples, annotations, splits, transforms, and provenance. Public data retains upstream license and hash references. Synthetic data records generator configuration. Private/local data remains outside Git and still needs provenance, access policy, and reproducible split hashes. Restricted or non-redistributable data must never be committed.

