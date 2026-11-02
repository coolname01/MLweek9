# STATE

Updated: 2026-11-02

## Context
Competition: sleep-edf-neuroscience-2026, edition full_psg_v1. Metric: macro-F1. The individual checkpoint was already submitted and defended.
Deadlines (MSK): team roster freeze 2026-11-04 14:00; final Kaggle solutions + Moodle package 2026-11-07 14:00; verified results by 2026-11-11 14:00. 3 submissions/day, up to 2 finals.

## Data facts (from competition page; verify locally)
- 152 records, 412,081 epochs: train 112 / 302,653; val 16 / 43,206; test 24 / 66,222.
- 3 channels (EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal), 100 Hz, 3000 samples, uV.
- Example record SC4011: 2,802 epochs, long wake before and after sleep.
- Private leaderboard: 6 subjects; small macro-F1 gaps may be noise.
- Class distribution (labels.csv): train W 208224, N1 16277, N2 51233, N3 8412, REM 18507; val W 28763, N1 1884, N2 7459, N3 1784, REM 3316.
- epoch_index gaps (step > 1): 142 gaps, 1104 missing 30 s slots. See docs/data_summary.md.

## Status
- [x] Task description and rules collected
- [x] full_psg_v1 counts match. SHA-256: 161/162 manifest files match; `archive/README.md` (2307 bytes) is absent; `team_info.csv` was not opened
- [x] Local folders: src, configs, notebooks, experiments, models, cache; .gitignore and .cursorignore
- [x] Data checks and class distribution (`docs/data_summary.md`)
- [ ] Push (local commit only)
- [x] Baseline: Welch PSD band features + amplitude stats + LightGBM, val macro-F1 0.7130
- [ ] Hypothesis 1: neighboring-epoch context within a record
- [ ] Final checkpoint + SHA-256, offline Kaggle notebook (<= 2 h)

## Results
`baseline_v1` val macro-F1 **0.7130** (W 0.9558, N1 0.3649, N2 0.7914, N3 0.7606, REM 0.6923), accuracy 0.8603. Control `always_W`: macro-F1 0.1599, accuracy 0.6657. Pooled over val epochs, not the mean of subjects.
Per-subject val macro-F1: SC404 0.7830, SC407 0.8437, SC417 0.6314, SC424 0.7514, SC426 0.7353, SC427 0.7195, SC457 0.5193, SC459 0.5316.
Log: `experiments/results.csv`. Model `models/baseline_v1.joblib` (not in git), SHA-256 `dde06e8a59ac81ad8669937f86c592756efde9c0526ac52a67ab4efa4e1f2685`. Notebook `notebooks/baseline.ipynb` reloads the cache and the model.

## Layout (actual, 2026-11-02)
- Signals: `data/train` 112 npz, `data/val` 16, `data/test` 24.
- `labels.csv`, `sample_submission.csv`, `team_info.csv`, `MANIFEST.json` and competition `*.md` are in `archive/`, not next to the splits.
- State file: `docs/STATE.md`.

## Next step
Hypothesis 1: neighboring-epoch context within a record. Keep this baseline frozen; do not refit it on val.

## Baseline run (2026-11-02)
LightGBM 4.7.0 had a cp314 wheel, so HistGradientBoosting was not used. `class_weight=balanced`, `n_estimators=200`, `deterministic=True`, seed 42. Fit on all 302,653 train epochs. Val was scored only. Test was not read.
Features (`cache/features_v1`, version `v1`): per-record per-channel z-score, then Welch (`fs=100`, `nperseg=256`, 50% overlap) log10 and relative power in delta 0.5–4, theta 4–8, alpha 8–12, sigma 12–16, beta 16–30 Hz, plus mean absolute value, std, skew, excess kurtosis, range, and Hjorth activity, mobility, complexity. 54 features. No parameter was fit across records.
Smoke, 2 train + 1 val, `n_jobs=10`: mean 4.00 s/record. Estimate `4.00 * 152 / 10` = 61 s. Peak private 731 MiB per worker, about 7.3 GiB at 10 workers; 16.9 GiB RAM free. Proceeded.
Full train+val extraction: wall 122.3 s (125 computed, 3 cache hits), mean 9.23 s/record at 10 workers, cache 94.8 MiB. Fit 13.8 s, val predict 0.3 s. `src.train` wall about 146 s.
Versions: Python 3.14.8, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, matplotlib 3.11.2, joblib 1.6.0, pyyaml 6.0.3. 12 logical CPUs, `n_jobs=10`. Windows-11-10.0.26300-SP0.

Checks ran on Python 3.14.8 (3.12 is not installed here), numpy 2.5.4, pandas 3.1.0, pyyaml 6.0.3, 58 s.

## Disclosures
- AI: Perplexity (planning, prompts, project files); Cursor (this baseline: features, training, notebook). Val metrics were recomputed from the feature cache and the saved model.
- External code / pretrained weights: none. LightGBM was trained here from the competition train split.
