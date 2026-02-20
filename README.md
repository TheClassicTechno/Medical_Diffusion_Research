# Medical_Diffusion_Research
part of CNS Lab
Juli Huang Work 


I train models to **predict the post-acetazolamide (ACZ) brain scan from the pre-ACZ scan**—so in principle a single acquisition could support CVR-style assessment without the drug challenge. This repo is a **unified pipeline**: same data, same split, same preprocessing and metrics for 2D and 3D models (conditional autoencoders, ResNet 3D, DDPM, Cold/Residual diffusion, FNO). Everything is designed for fair comparison and reproducibility.

---

## In short

- **Data:** 252 scans (2020–2023), subject-level split 189 train / 31 val / 32 test. MNI 91×109×91, brain mask, min-max to [0,1].
- **Metrics:** MAE, SSIM, PSNR inside the brain only. Lower MAE is better; higher SSIM and PSNR are better.
- **Protocol:** Three seeds (42, 123, 456), early stopping, mean ± std reported. Phase 2 adds vascular region-weighted loss (3D masks, no 2D slice hacks).

---

## Best results (from our paper)

**2D baseline (middle slice):** MAE 0.0497, SSIM 0.7886, PSNR 21.49 dB.

**Reproducible (three-seed) best:**

| Model   | MAE      | SSIM   | PSNR (dB) |
|--------|----------|--------|-----------|
| CAE_3D_s (script 3D) | 0.0689 ± 0.0008 | 0.7971 ± 0.0017 | **23.73 ± 0.09** |
| CAE_3D (external)    | 0.0742 ± 0.0039 | 0.7909 ± 0.0039 | 23.10 ± 0.43 |

**Extended (single-run) best — full-volume 3D beats 2D on all three metrics:**

| Model                      | MAE     | SSIM   | PSNR (dB) |
|---------------------------|---------|--------|-----------|
| **Residual Diffusion 3D (tips)** | **0.0228** | **0.8528** | **26.13** |
| CAE_3D                    | 0.0253  | 0.8513 | 25.27     |
| FNO 3D                    | 0.0301  | 0.696  | 25.58     |

So: **best MAE 0.0228, best SSIM 0.85, best PSNR 26.13 dB**—all from 3D models that outperform the 2D baseline. We also report per-region MAE in vascular territories, Bland–Altman limits, and pairwise significance across models.
