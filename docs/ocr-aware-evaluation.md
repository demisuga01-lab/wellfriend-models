# OCR-aware evaluation hooks

MP6 adds observation-based hooks for `cer_delta_after_restoration`, `wer_delta_after_restoration`, text-edge preservation, foreground-stroke preservation, and binarization text contrast. They consume supplied text/edge/stroke observations and do not import, bundle, or imply an OCR engine.

An OCR improvement claim requires a named OCR runtime, normalization policy, licensed evaluation data, language/script, and reproducible report in a later phase.
