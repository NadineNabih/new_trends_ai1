# EEG representation learning on ZuCo 1.0 / 2.0 — EEG-to-sentence decoding

## Status (read this first)
* The code runs end-to-end on **synthetic files in the ZuCo v7.3 layout** (loader, caching, splits, SSL, decoder,
  controls, robustness, cross-dataset, plots). It has **not** been run on the real ZuCo data, so there are **no real results yet**.
* Real-data risks: (1) field layout of the Kaggle `.mat` files — run `python run.py inspect <file.mat>` first;
  (2) `CBraModWrapper` is untested against the CBraMod repo; (3) hyper-parameters are untuned defaults.

## Run (Kaggle notebook with GPU is the easiest place)
```bash
pip install -r requirements.txt
python run.py inspect /kaggle/input/zuco-1-full-matlab/<any>/resultsZAB_NR.mat      # sanity check of the structure
python run.py prepare --src /kaggle/input/zuco-1-full-matlab --dataset zuco1
python run.py prepare --src /kaggle/input/zuco-2-full-matlab --dataset zuco2
python run.py run --exp all --datasets zuco1 zuco2                  # add --tasks NR to use one task only
# optional pretrained model:  --cbramod_repo /path/CBraMod --cbramod_weights /path/pretrained_weights.pth
python run.py plots
```
Outputs (`results/`): `representations_*.csv`, `robustness_*.csv`, `quality_vs_performance_*.csv`, `cross_*.csv`, figures.

## What maps to which project step
| Step | Where |
|---|---|
| 1 data + subject-held-out splits | `data.make_splits`: subjects **and** sentences held out independently |
| 2 preprocessing | `data.preprocess`: demean, 50 Hz notch, 0.5–45 Hz Butterworth, resample 500→128 Hz, crop/pad 12 s, per-trial channel z-score |
| 3 representations (≥2, one self-supervised on ZuCo) | `experiments.RECIPES`: band-power baseline, random frozen, **SSL(ZuCo) frozen / fine-tuned**, scratch, optional pretrained CBraMod |
| 4 EEG→language | contrastive (InfoNCE) alignment to frozen sentence embeddings; retrieval top-1/5/10, MRR, cosine gap, RSA, permutation p |
| 4 noise control | noise at test; model *trained* on noise (same length/scale) |
| 5 quality + robustness | `quality.py` degradations (gauss, dead channels, 50 Hz, artifact bursts); quality-aware FiLM + degradation augmentation vs plain |
| 6 generalisation | unseen-subject splits; `exp_cross` zero-shot and SSL-transfer ZuCo1↔ZuCo2 |
| 7 analysis | CSVs + figures |

## Design decisions that matter for validity
* **Sentence leakage.** Subjects read the same sentences, so a subject-only split lets the decoder memorise the text side.
  The headline number is `split=test` (unseen subject *and* unseen sentence); `test_seen_sent` is reported to show the gap.
* **Duration confound.** Longer sentences produce longer recordings. Noise controls keep the true length and amplitude, so
  they measure what duration alone can do; EEG content only counts above that.
* **Cross-dataset overlap.** Sentences of the target dataset that also occur in the source dataset are dropped from cross tests.
* SSL pre-training uses train subjects/sentences only. Candidate pool for retrieval = unique sentences of the evaluated split
  (chance is reported next to every number).
* Cross-dataset runs need equal channel counts (the first `min(C1,C2)` channels are used otherwise — check the montage match).
