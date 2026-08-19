# Pruning

Pruning plans support unstructured, structured, and channel-placeholder modes. Every report must contain target and achieved sparsity plus a metric-delta object; MP6’s no-weight path reports `metadata_only` and zero achieved sparsity rather than inventing an optimization result.

Magnitude/channel pruning for actual tensor models is optional follow-up work behind the `torch` extra.
