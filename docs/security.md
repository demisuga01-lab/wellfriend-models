# Security and governance risks

Dataset and artifact manifests are untrusted structured input: validators require schema fields, hashes, task enums, licenses, provenance, and promotion evidence. Production code must never import arbitrary Python or execute an unvalidated artifact. Restricted, unknown, research-only, and non-commercial data/weights are blocked from production artifacts.
