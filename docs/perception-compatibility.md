# Perception compatibility

MP6 artifacts retain the task names expected by `wellfriend-perception`: `document_segmentation`, `document_corner_regression`, `document_quality`, `document_cleanup_mask`, and `document_restoration`. Image schemas declare `Gray8` or RGB-like inputs compatible with perception’s `Gray8`, `Rgb8`, and `Bgr8` conversion boundary. Corner outputs remain ordered TL/TR/BR/BL in image pixel coordinates; confidence remains a bounded runtime concern.

Device classes match perception routing: low, mid, high, server, web, unknown. Restoration planning keeps processor IDs (`brightness_contrast`, `gamma`, `denoise`, `unsharp`, `background_normalization`, `grayscale`, `binarize`) and filter names (Original, Auto, Clean, Color, Grayscale, B&W, Receipt, Book, Whiteboard, PhotoDocument) as runtime-owned identifiers. Model metadata adds tiling policy only; tiling execution remains a future adapter responsibility.

No hard mismatch was found, so `wellfriend-perception` was not modified. Its adapter must still reject missing weights/checksums and never import this Python package.
