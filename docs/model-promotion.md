# Model promotion gates

| Status | Meaning |
| --- | --- |
| `placeholder` | No weights; cannot enter production. |
| `experimental` | Synthetic/tiny-data exploration; no quality claim. |
| `research` | Data/license evidence retained; not a release. |
| `candidate` | Reproducible evaluation evidence required. |
| `mobile_candidate` | Export, size, device profile, runtime plan, metrics, and limitations required. |
| `production_ready` | Requires safe data/weights, hashes, metrics, runtime/provenance/safety evidence. |
| `deprecated` | Retained only for migration. |
| `blocked` | Cannot be used. |

MP6 contains placeholders and experimental entries only. No artifact is marked `production_ready`.
