# STATE

Updated: 2026-11-03

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
- [x] Hypothesis 1: neighboring-epoch context within a record (`context2_v1`, criterion passed)
- [ ] Final checkpoint + SHA-256, offline Kaggle notebook (<= 2 h)

## Results
`baseline_v1` val macro-F1 **0.7130** (W 0.9558, N1 0.3649, N2 0.7914, N3 0.7606, REM 0.6923), accuracy 0.8603. Control `always_W`: macro-F1 0.1599, accuracy 0.6657. Pooled over val epochs, not the mean of subjects.
Per-subject val macro-F1: SC404 0.7830, SC407 0.8437, SC417 0.6314, SC424 0.7514, SC426 0.7353, SC427 0.7195, SC457 0.5193, SC459 0.5316.
Log: `experiments/results.csv`. Model `models/baseline_v1.joblib` (not in git), SHA-256 `dde06e8a59ac81ad8669937f86c592756efde9c0526ac52a67ab4efa4e1f2685`. Notebook `notebooks/baseline.ipynb` reloads the cache and the model.

`context2_v1` val macro-F1 **0.7597** (exact 0.759678; W 0.9692, N1 0.4418, N2 0.8151, N3 0.7797, REM 0.7926), accuracy 0.8899. Delta vs `baseline_v1`: macro-F1 +0.0467, N1 +0.0769, REM +0.1003. Acceptance was macro-F1 >= 0.723 and a gain on at least 5 of 8 val subjects. Result: **passed** (8/8 subjects improved). Pooled over val epochs. Test was not read. `cache/features_v1` was not recomputed. `baseline_v1.joblib` was not rewritten (SHA-256 unchanged).
Per-subject val macro-F1 (`baseline_v1` -> `context2_v1`): SC404 0.7830 -> 0.8064 (+0.0234), SC407 0.8437 -> 0.8699 (+0.0261), SC417 0.6314 -> 0.6785 (+0.0471), SC424 0.7514 -> 0.7773 (+0.0259), SC426 0.7353 -> 0.7801 (+0.0448), SC427 0.7195 -> 0.7674 (+0.0479), SC457 0.5193 -> 0.5716 (+0.0523), SC459 0.5316 -> 0.6109 (+0.0793).
Model `models/context2_v1.joblib` (not in git), SHA-256 `ef7125020f85b4fdbdedb5c28f705fe3695b8ca33d3ba688f56576df41b5e4bd`.

## Layout (actual, 2026-11-02)
- Signals: `data/train` 112 npz, `data/val` 16, `data/test` 24.
- `labels.csv`, `sample_submission.csv`, `team_info.csv`, `MANIFEST.json` and competition `*.md` are in `archive/`, not next to the splits.
- State file: `docs/STATE.md`.

## Next step
`context2_v1` is the accepted finalist: val macro-F1 0.759678, stable across seeds 42/7/123, deterministic, Kaggle estimate 0.855 min. Do not refit it on val or test. E6–E10 were not started.

## E3 context ±4 (2026-11-03)
Same LightGBM settings and the same `features_v1` cache. Offsets -4,-3,-2,-1,+1,+2,+3,+4, plus `n_neighbors` 0..8. 487 float32 columns. Train matrix 0.549 GiB.
Smoke `SC4011`, `SC4012`, `SC4041`: shapes (5650, 487) and (2569, 487). NaN fraction 0.000785 / 0.001209. On `SC4041` row 1365, `epoch_index` 1365, next file row 1367: offset +1 is NaN and offset +2 is 1367. Stacking the two train records was refused. Fit estimate 11.34 min. Actual context build 1.71 s, fit 140.1 s, predict 0.88 s.
Val macro-F1 **0.765883** (W 0.9708, N1 0.4512, N2 0.8194, N3 0.7746, REM 0.8134), accuracy 0.8946. Delta vs `context2_v1` +0.006205. Subjects improved 6/8. Gate was macro-F1 >= 0.769678 and 5/8 subjects. **Rejected** (subject gate passed, macro gate did not). Model `models/context4_v1.joblib`, SHA-256 `9ec8e4022b4ef5b5b4bfd47a9dc0251caa77de01f8e9db90c11a2c18d27eedb0`. Rebuilt ±2 features reproduced `context2_v1` macro-F1 0.759678.

## E4 features v2 (2026-11-03)
16 fixed columns added to the current epoch of `context2_v1` (287 columns). Cache `cache/features_v2`, per-record z-score, Welch grid identical to v1. Columns: normalized spectral entropy 0.5–50 Hz per channel; EEG delta/beta and theta/alpha; linear EEG sigma power 12–16 Hz; EOG log10 power 0.5–5 Hz; mean and max variance of 15 non-overlapping 2 s blocks per channel. Sigma linear power is monotone with v1 log-sigma and was kept because the queue names that band.
Smoke mean 1.57 s/record. Extract estimate 0.33 min; actual extract wall 87.3 s (slower than the linear estimate, still inside 2× the combined extract+fit estimate). Fit estimate 6.62 min; actual fit 73.4 s. Train matrix 0.324 GiB.
Val macro-F1 **0.754609** (W 0.9688, N1 0.4336, N2 0.8071, N3 0.7777, REM 0.7858), accuracy 0.8871. Delta vs `context2_v1` -0.005069. Subjects improved 2/8. **Rejected**. The nested context block still reproduced 0.759678. Model `models/v2_context2_v1.joblib`, SHA-256 `7faac754f9364e3a67d2d05f233f1fcc1364028de49d06ffad2b029ad185b58f`. Accepted best remains `context2_v1`.

## E5 rolling windows (2026-11-03)
Centered rolling mean and population std of the 54 v1 columns, windows of 5 and 11 epochs, concatenated onto `context2_v1`. A window stops where `epoch_index` does not increase by 1. On `SC4041` row 1365 the 5-epoch window mean of `epoch_index` is 1364 and does not include the next file row 1367. 487 columns, train matrix 0.549 GiB.
First smoke fit was 49.9 s and the linear extrapolation was 44.5 min, so that attempt stopped before the full fit. A rerun of the same smoke fit was 12.3 s (linear estimate 11.0 min). The same-width `context4_v1` fit had been 140 s against an 11.3 min linear estimate, so the full fit was allowed with a 40 min cap. Actual matrix build 2.83 s, fit 218.6 s, predict 0.82 s.
Val macro-F1 **0.766738** (W 0.9735, N1 0.4421, N2 0.8171, N3 0.7713, REM 0.8297), accuracy 0.8969. Delta vs `context2_v1` +0.007059. Subjects improved 3/8. **Rejected**. Model `models/smooth_v1.joblib`, SHA-256 `af40100bb95bfecc5a511601bbec887d2697c2d4eb82645a95e95d8974c5a8a3`.

## Queue stop
E3, E4 and E5 were rejected one after another. E6, E7, E8, E9 and E10 were not started. Torch is not installed, so E9 would also have been skipped for lack of a GPU. The accepted model remains `context2_v1`.

## Confirmation of context2_v1 (2026-11-03)
Seeds on the same train matrix, val macro-F1: seed 42 **0.759678**, seed 7 **0.757112**, seed 123 **0.759967**. Mean 0.758919, min 0.757112, max 0.759967. max-min = 0.002855, below 0.01, so the result is stable.
A second seed-42 training wrote `models/context2_seed42_repeat.joblib` and produced the same val `id,label` SHA-256 as the original model: `466df915f39120b6b7383937ed88913ef15434fd6188a3d1d89261e1278f983e`. Two consecutive CPU inference passes also matched. `deterministic=yes`.
Kaggle-style budget, test not read. Feature time on val record `SC4041`, one process, one thread: 1.972 s, times 24 = 47.3 s. Model load 0.031 s. Val predict with 4 threads 0.858 s, scaled to 66,222 epochs 1.314 s. GPU inference was not available. Peak private RAM for one-record inference 0.591 GiB. `est_kaggle_minutes = (load + features_24 + 3 * scaled_predict) / 60 = 0.855`. The separate 24-record inference estimate is 0.022 min. Both are inside 60 min and 30 min.
Val probabilities: `models/context2_v1_val_proba.npy`. Feature list (271 names): `experiments/feature_list_context2_v1.txt`. Val `id,label`: `experiments/context2_v1_val_id_label.csv`. The original `models/context2_v1.joblib` was not overwritten. SHA-256 `ef7125020f85b4fdbdedb5c28f705fe3695b8ca33d3ba688f56576df41b5e4bd`.

## Baseline run (2026-11-02)
LightGBM 4.7.0 had a cp314 wheel, so HistGradientBoosting was not used. `class_weight=balanced`, `n_estimators=200`, `deterministic=True`, seed 42. Fit on all 302,653 train epochs. Val was scored only. Test was not read.
Features (`cache/features_v1`, version `v1`): per-record per-channel z-score, then Welch (`fs=100`, `nperseg=256`, 50% overlap) log10 and relative power in delta 0.5–4, theta 4–8, alpha 8–12, sigma 12–16, beta 16–30 Hz, plus mean absolute value, std, skew, excess kurtosis, range, and Hjorth activity, mobility, complexity. 54 features. No parameter was fit across records.
Smoke, 2 train + 1 val, `n_jobs=10`: mean 4.00 s/record. Estimate `4.00 * 152 / 10` = 61 s. Peak private 731 MiB per worker, about 7.3 GiB at 10 workers; 16.9 GiB RAM free. Proceeded.
Full train+val extraction: wall 122.3 s (125 computed, 3 cache hits), mean 9.23 s/record at 10 workers, cache 94.8 MiB. Fit 13.8 s, val predict 0.3 s. `src.train` wall about 146 s.
Versions: Python 3.14.8, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, matplotlib 3.11.2, joblib 1.6.0, pyyaml 6.0.3. 12 logical CPUs, `n_jobs=10`. Windows-11-10.0.26300-SP0.

Checks ran on Python 3.14.8 (3.12 is not installed here), numpy 2.5.4, pandas 3.1.0, pyyaml 6.0.3, 58 s.

## Context run (2026-11-03)
Same LightGBM settings as `baseline_v1` (`n_estimators=200`, `learning_rate=0.1`, `num_leaves=31`, `class_weight=balanced`, `deterministic=True`, seed 42, `n_jobs=10`). Features: the 54 cached `v1` columns plus the same columns at `epoch_index` offsets -2, -1, +1, +2 inside the recording, plus `n_neighbors` (0..4). A neighbor is NaN unless that index exists in the same recording. 271 float32 columns. Full train matrix 0.306 GiB.
Smoke, 2 train + 1 val (`SC4011`, `SC4012`, `SC4041`): train shape (5650, 271), val (2569, 271). NaN fraction 0.000423 train / 0.000776 val. `SC4041` row 1365 has `epoch_index` 1365 and the next file row is 1367; offset +1 is NaN and offset +2 is 1367. Stacking `SC4011` with `SC4012` was refused (duplicate `epoch_index`). Scaled fit estimate 6.78 min, under the 20 min / 8 GiB gate.
Full run: context build 2.55 s, fit 108.6 s, val predict 0.56 s. Reloaded baseline macro-F1 0.712991, matching the log. Versions printed for this process: numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, joblib 1.6.0. LightGBM is the 4.7.0 wheel from the baseline run.

## Disclosures
- AI: Perplexity (planning, prompts, project files); Cursor (baseline: features, training, notebook; context2: neighbor features and training). Val metrics for `context2_v1` were computed in this run. The reloaded `baseline_v1` score matched the log.
- External code / pretrained weights: none. LightGBM was trained here from the competition train split.
