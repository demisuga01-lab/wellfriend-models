# Distillation

The distillation contract supports torch-model, classical/scalar, artifact, cached-prediction, and external audit-gated teachers. It records teacher/student identities, task, loss selection, data-manifest hash, reproducibility metadata, loss curves, metrics, and teacher/student delta.

An external teacher such as DocRes must carry `audit-gated` status. That status cannot promote its student to production. MP6 uses a scalar synthetic smoke path only.
