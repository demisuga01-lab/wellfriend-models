# Evaluation metrics

MP5 implements binary IoU, Dice, mask precision/recall/F1, corner MAE/RMSE/normalized error, deterministic polygon-IoU approximation, CER, WER, PSNR, global SSIM, latency summaries, and artifact file size. CER/WER operate on strings and do not require an OCR runtime.

MS-SSIM, hardware memory, FLOPs/MACs, and full geometric-distortion analysis have explicit placeholder seams. Global SSIM is a scalar baseline rather than a substitute for windowed or multiscale perceptual validation. Metrics must identify the evaluation data and should not be compared across differently licensed or differently transformed data without disclosure.
