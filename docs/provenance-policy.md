# Provenance policy

Every experiment preserves the exact config, current Git commit, portable environment/dependency summary, synthetic/dataset manifest hash, metrics, and event log. Every artifact records source, data references, code license, weights license, limitations, safety notes, and checksums.

Do not replace an artifact in place. Publish a new semantic version with new checksums. Do not infer a weight’s license from an adjacent repository license. A missing source, dataset citation, or license record is a release blocker.
