# ADR-0001: Model research is separated from production runtime

Python training/evaluation remains in this repository. Production perception remains a separately versioned Rust runtime and consumes validated artifact files rather than Python imports.
