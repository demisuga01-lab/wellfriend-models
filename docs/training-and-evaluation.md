# Training, evaluation, and export

MP5 provides deterministic synthetic CPU smoke entry points for document segmentation, corner regression, and quality. The same contracts reserve restoration, cleanup masks, and OCR benchmarking without claiming their implementation or quality.

Evaluation implements IoU, Dice, corner error, CER, WER, PSNR, global SSIM, latency summary, and model-size helpers. MS-SSIM, hardware memory, FLOPs/MACs, and production geometric evaluation remain clearly marked placeholders. Export currently emits and validates no-weights placeholder artifacts; ONNX export is an audit-gated optional follow-up.
