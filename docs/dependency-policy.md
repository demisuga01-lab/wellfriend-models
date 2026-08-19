# Dependency and license policy

Every direct Python/build/CI dependency is registered with version constraint, license, source URL, purpose, risk level, consumer, and scope. `tools/check_dependency_register.py` rejects an incomplete register or GPL-family, non-commercial, research-only, or unknown direct dependency license.

Datasets, weights, and third-party training code have separate provenance/license gates. A permissive code license does not authorize model weights or training data. MP5 contains neither external datasets nor third-party weights/code.
