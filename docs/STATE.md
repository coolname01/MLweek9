# STATE

Updated: 2026-11-04

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
- [x] Hypothesis C: per-record standardization of `features_v1` (`record_z_c1`, criterion passed)
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
`record_z_c1` is the accepted model: val macro-F1 0.774429, stable across seeds 42/7/123, deterministic, Kaggle estimate 0.920 min. It replaces `context2_v1` (0.759678). Do not refit it on val or test. The offline notebook is still open. Hypotheses A and B were not run.

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

## E6 probability stack (2026-11-03)
Stage 1 uses the context2 matrix (271 columns). Train probabilities are GroupKFold by `subject_id`, 5 folds, 58 subjects. Stage 2 is LightGBM on the five class probabilities at `epoch_index` offsets -3..+3 inside the same record (35 columns). A missing index is NaN. Val probabilities come from the saved full-train `context2_v1` model. That model's val probabilities matched `models/context2_v1_val_proba.npy` with max abs 0, and the model file hash was unchanged. Test was not read. `champion.json` was not replaced.

Smoke on `SC4041` row 1365: offset +1 is NaN and offset +2 is epoch 1367. Gluing `SC4011` with `SC4012` was refused. The 7.24 min parent-fit estimate was not a stop. Actual OOF 326.3 s (folds 67.8, 62.8, 63.9, 63.3, 64.7 s), stage-2 fit 8.4 s. Peak training working set 1.70 GiB.

Val macro-F1 **0.739997** (W 0.9606, N1 0.4521, N2 0.7818, N3 0.7669, REM 0.7385), accuracy 0.8715. Delta vs `context2_v1` -0.019681. Subjects improved 2/8. **Rejected**. SC457 0.539573, SC459 0.605336.

Model `models/stack_v1.joblib`, SHA-256 `7774504ffb339cd57bc1edec28b80e2439075dfe3dbdbee2f322bd1633fee7d5`. Two consecutive val `id,label` files matched, `deterministic=yes`. `est_kaggle_minutes=1.010`, `ram_peak_gb=0.585`. Feature time on `SC4041` 2.102 s (24 records 50.4 s), model load 0.059 s, CPU predict with 4 threads 2.203 s. Torch is not installed, so GPU inference was not measured.

## E7 class weights (2026-11-03)
Two variants only, both on the context2 matrix. Balanced train weights are W 0.290699, N1 3.718781, N2 1.181477, N3 7.195744, REM 3.270687. The largest is 7.196, so a cap of 10 does not bind. The clipped weights were still passed as an explicit dictionary and the model was trained. Val macro-F1 **0.759678**, the same per-class F1 as `context2_v1`, and the same macro-F1 on all 8 subjects. Delta 0. Subjects improved 0/8. **Rejected**. Fit 82.7 s. `deterministic=yes`. `est_kaggle_minutes=0.909`. `ram_peak_gb=0.582`. SC457 0.571615, SC459 0.610868. Model `models/classweight_cap10_v1.joblib`, SHA-256 `9a020f83ed8cc3b4e33a5c2970fa47da296d0413244984421252ec317b0755f2`. Test was not read. `champion.json` was not replaced.

## E8 LightGBM hyperparameters (2026-11-03)
Six settings: `num_leaves` 31/63/127 and `learning_rate` 0.05/0.1. `n_estimators` is the early-stopping round (patience 50, cap 600) on macro-F1 of a GroupShuffleSplit holdout of 12 train subjects (SC401, SC405, SC408, SC415, SC416, SC438, SC450, SC454, SC465, SC467, SC471, SC475; 65,669 epochs). The choice was written to `experiments/e8_choice.json` before val features were loaded. Val was scored only for the winner. Test was not read.

Internal macro-F1: 31/0.05 = 0.787390 (286 trees), 31/0.1 = 0.787492 (141), 63/0.05 = 0.787805 (166), 63/0.1 = 0.788638 (199), 127/0.05 = 0.789636 (206), 127/0.1 = 0.790380 (190). The last of these was selected. Final fit on all train took 137.8 s.

Val macro-F1 **0.759073** (W 0.9731, N1 0.4454, N2 0.8254, N3 0.7462, REM 0.8053), accuracy 0.8983. Delta vs `context2_v1` -0.000606. Subjects improved 3/8. **Rejected**. SC457 0.565652, SC459 0.595608. `deterministic=yes`. `est_kaggle_minutes=0.920`. `ram_peak_gb=0.589`. Model `models/lgbm_internal_v1.joblib`, SHA-256 `17ce051473f1d75740cfae766f615b6c51c73c35b9c1d90613e41567d57d196a`. `champion.json` was not replaced.

## E9 1D CNN (2026-11-03)
Torch 2.14.1+cu126, CUDA available on the RTX 3060. Driver 617.42, CUDA UMD 13.4. Python 3.14.8 has this wheel; no Python 3.12/3.13 is installed. Versions are in `requirements_torch.txt`. A fresh cu128 download from the official index stayed near 65 KB/s, so the environment received the official cu126 build already present on this machine.

Input: per-record per-channel z-score (population std, floor 1e-6), float16 memmap `cache/raw_f16`. Three epochs t-1, t, t+1 of the same record, zeros and a mask when `epoch_index` has no exact neighbor. `SC4022` and `SC4041` gaps keep offset +1 missing. Test was not read. 457,957 parameters, 5 conv blocks, AdamW, OneCycle max_lr 0.001, autocast, batch 128. Smoke: 0.215 s/batch, reserved VRAM 0.44 GiB. The 397 s epoch estimate was not a stop. Actual epoch 1 was 86.4 s.

Epoch choice used a 12-subject holdout of train (46/12, same seed 42 as E8). Internal macro-F1: epoch 1 0.707438, epoch 2 0.664845, epoch 3 0.662945, epoch 4 0.670556. Patience 3 kept epoch 1. `experiments/e9_choice.json` was written before the val cache. Val cache then took 18.7 s.

Val macro-F1 **0.675614** (W 0.9632, N1 0.3478, N2 0.7218, N3 0.7458, REM 0.5994), accuracy 0.8364. Delta vs `context2_v1` -0.084064. Subjects improved 0/8. **Rejected**. SC457 0.438812, SC459 0.468528. Train 362.5 s. Two GPU val passes matched, `deterministic=yes`. CPU labels differed from GPU. `est_kaggle_minutes=6.594` on CPU (4 threads) and 1.315 on GPU. `ram_peak_gb=1.205`. Model `models/cnn_raw3_v1.pt`, SHA-256 `1798e641762f9274d8bb3cff9d2c0021b6044bd824fc1101935922aa9e7d568d`. `champion.json` was not replaced.

## E10 blend (2026-11-03)
No E6–E8 model was accepted, so the tabular component is `context2_v1`. The CNN was trained, so the blend is context2 plus `cnn_raw3_v1`. Weights are only (1, 0.5) and (1, 1). Probabilities were aligned by `id`. Test was not read. `champion.json` was not replaced.

`ensemble_w1_05` val macro-F1 **0.761842** (W 0.9700, N1 0.4393, N2 0.8163, N3 0.7865, REM 0.7971), accuracy 0.8894. Delta +0.002163. Subjects improved 4/8. **Rejected**. SC457 0.560820, SC459 0.615302. SHA-256 of the spec file `d0c1d1b2cda9230bdc8f7732755039c1f3ccb0b79c867313820198077b8c3da1`.

`ensemble_w1_1` val macro-F1 **0.755558** (W 0.9698, N1 0.4289, N2 0.8067, N3 0.7865, REM 0.7858), accuracy 0.8841. Delta -0.004120. Subjects improved 4/8. **Rejected**. SC457 0.531954, SC459 0.611934. SHA-256 `72161051c4c71a4881b875cc0b1dababfb90d1107045ffe45f26f0b019024f2f`.

Both blends: `deterministic=yes`, `est_kaggle_minutes=6.800` on CPU and 1.966 on GPU, `ram_peak_gb=1.254`.

## Queue
E6–E10 are finished and rejected. Experiment C accepted `record_z_c1`. Hypotheses A and B were not run.

## Error analysis of context2_v1 (2026-11-03, no training)
Sources: `models/context2_v1_val_proba.npy`, `experiments/context2_v1_val_id_label.csv`, `archive/labels.csv`, val part of `cache/features_v1`. Test was not read. Scripts live outside the repository.
- 4755 val errors. 51% lie within 1 epoch of a true stage change (error rate 45% at distance 0, 1.5% beyond 30 epochs). 57% of errors sit in runs of >= 4 consecutive wrong epochs, isolated errors are 18%. Median max-probability 0.67 on errors vs 0.999 on correct epochs.
- Time from record start: 0 errors in hour 0-1, 0.2% in 1-2 h, 78% of errors at 8-16 h. Sleep onset varies 2.0-11.3 h across val records. First/last 20 file rows: 0 errors. `n_neighbors` = 3: 32 errors of 108 epochs (0.7% of all errors); = 2: 0 of 36.
- Confusions: W->N1 989 (21%, 82% inside the sleep period), N2->N1 837, N3->N2 478 (N3 recall 71%), N2->REM 387, REM->N1 387, N1->N2 346. N1 precision 0.34, recall 0.62 (3381 predicted vs 1884 true).
- Diagnostic only: averaging the saved probabilities over +-k epochs lowers macro-F1 monotonically (+-1 0.7590, +-5 0.7546, +-20 0.7064).
- SC457: `SC4571` has no N3, `SC4572` has 29 N3 epochs, 26 of them predicted W (pW 0.5-0.9, pN3 <= 0.04) inside N2/N3 context, so subject N3 F1 = 0 and the ceiling without N3 is about 0.77. N2 recall 58% (N2->N1 22%, N2->REM 10%). 18.6 transitions per hour of sleep period (highest, with SC459).
- SC459: amplitude drift and artifacts. W-epoch EEG std first third 0.55/0.66 vs last third 1.26/1.85 (`SC4591`/`SC4592`), corr(log std, time) 0.52/0.30, 2.0%/6.0% epochs with kurtosis > 10. Record mean of Fpz-Cz delta log-power -1.23 (`SC4591`) vs train -0.63 +- 0.14; N3 median -1.02/-0.77 vs train median -0.30. After per-record standardization N3 of `SC4591` is at z = 0.43, inside the train IQR 0.33-0.96. N3 recall 19% (81% -> N2), N1 recall 38%, REM->N2 21%.
- Rules: `archive/rules.md` section 4 and `archive/academic_integrity.md` explicitly allow normalization by the record's own mean and std under a fixed formula; `src/features.py::_zscore_record` already uses whole-record statistics.

Experiment C was run next. The formula was fixed before the fit. See the section below.

## E11 per-record feature standardization (2026-11-04)
Frozen formula, one recording at a time: `z = (x - mean_rec) / (population_std_rec + 1e-6)`. `population_std_rec` is ddof 0, `sqrt(mean((x - mean)^2))`, over every epoch of that recording and over no other recording. The same ±2 context is then built from `z` (`epoch_index` offsets -2, -1, +1, +2; missing indexes are NaN; `n_neighbors` is 0..4). LightGBM matches `context2_v1`: `n_estimators=200`, `learning_rate=0.1`, `num_leaves=31`, `class_weight=balanced`, `deterministic=True`, `n_jobs=10`, seed 42. Two variants only. C1 is the z context, 271 columns. C2 is the raw context body (270), the z context body (270) and one `n_neighbors`, 541 columns. The gate was val macro-F1 >= 0.769678 and a strict gain on at least 5 of 8 val subjects. If both passed and C2 did not lead by more than 0.005, C1 was selected. Test was not read. `cache/features_v1` was not written; its tree SHA-256 stayed `93c72908dcb8cba0725be8eeb59dbbd42bd0f69fbfb7324a56ba0e812550a4bc` (128 files). `models/context2_v1.joblib` stayed `ef7125020f85b4fdbdedb5c28f705fe3695b8ca33d3ba688f56576df41b5e4bd`. The 54 z columns are in `cache/features_c1` (112 train + 16 val). Rebuilding the raw ±2 matrix reproduced all 43,206 `context2_v1` val labels before either fit.

`zscore_record_features` takes one array. `transform_record(features, epoch_index, variant)` takes one recording. Pooling `SC4011` with `SC4012` changed the `SC4011` z values by max abs 1.764, so a test recording can be standardized without any other recording.

Smoke `SC4011`, `SC4012`, `SC4041`: shapes (2802, 271)/(2802, 541), (2848, 271)/(2848, 541), (2569, 271)/(2569, 541). Z NaN and inf fractions were 0. Context NaN fractions were 0.000427 / 0.000420 / 0.000776 for C1 and 0.000427 / 0.000421 / 0.000777 for C2. Constant columns (population std 0) were 0 on all three. Matrix time 0.009 / 0.010 / 0.009 s. On `SC4041` the first gap is epoch 1365; z-context sources are 1363, 1364, missing, 1367, and offset +1 is not the next file row. Smoke wall 0.596 s. Cache write 1.47 s.

`record_z_c1` val macro-F1 **0.774429** (exact 0.7744287626769817; W 0.975928, N1 0.465446, N2 0.815067, N3 0.767074, REM 0.848628), accuracy 0.9000. Delta vs `context2_v1` +0.014750. N1 precision 0.359078, recall 0.661359. Subjects improved 6/8. **Accepted**. Fit 68.8 s, predict 0.84 s. SC457 0.637965, SC459 0.610518.
Per-subject val macro-F1 (`context2_v1` -> C1): SC404 0.8064 -> 0.7903 (-0.0161), SC407 0.8699 -> 0.8739 (+0.0040), SC417 0.6785 -> 0.7265 (+0.0480), SC424 0.7773 -> 0.8071 (+0.0298), SC426 0.7801 -> 0.7991 (+0.0190), SC427 0.7674 -> 0.7723 (+0.0049), SC457 0.5716 -> 0.6380 (+0.0664), SC459 0.6109 -> 0.6105 (-0.0004).
Model `models/record_z_c1.joblib`, SHA-256 `3f5bfa834b9bac22c641acbb96a2ca27e9f8b642347fd600afe7845e718ed3b8`.

`record_z_c2` val macro-F1 **0.776073** (W 0.977030, N1 0.458030, N2 0.829205, N3 0.768897, REM 0.847202), accuracy 0.9036. Delta vs `context2_v1` +0.016394. N1 precision 0.363722, recall 0.618365. Subjects improved 7/8. The gate passed. C2 minus C1 is +0.001644, which is within 0.005, so C1 was kept. Decision `not_selected`. Fit 151.3 s, predict 0.89 s. SC457 0.605079, SC459 0.596575.
Per-subject val macro-F1 (`context2_v1` -> C2): SC404 0.8064 -> 0.8072 (+0.0008), SC407 0.8699 -> 0.8770 (+0.0071), SC417 0.6785 -> 0.7219 (+0.0434), SC424 0.7773 -> 0.7949 (+0.0176), SC426 0.7801 -> 0.7959 (+0.0158), SC427 0.7674 -> 0.7837 (+0.0163), SC457 0.5716 -> 0.6051 (+0.0335), SC459 0.6109 -> 0.5966 (-0.0143).
Model `models/record_z_c2.joblib`, SHA-256 `f99bfb8b77f43eba9df6dbf830d1d421073f13cb1e01667210ae337b251d3a6b`.

Seeds of `record_z_c1`, same train matrix: seed 42 **0.774429**, seed 7 **0.771845**, seed 123 **0.772182**. Mean 0.772819, min 0.771845, max 0.774429. max-min = 0.002584, below 0.01, so the result is stable. A second seed-42 fit produced the same val `id,label` SHA-256 as the original scoring pass and as a second predict pass: `93d82d0ccaf547e0d3bc0ad36b3e419e8e50a584e4440c4a797027b5d633fdd4`. `deterministic=yes`.
Kaggle-style budget, test not read. Feature time on val record `SC4041`, one thread, from the signal through `features_v1` and the C1 z context: 2.119 s, times 24 = 50.9 s. Model load 0.036 s. Val predict with 4 threads 0.944 s, scaled to 66,222 epochs and counted three times. `est_kaggle_minutes = 0.920`. Peak private RAM of a fresh one-record C1 predict: 0.585 GiB. Both are inside 60 min. GPU inference was not used (`lgbm_cpu_only`).
Confirmation models: seed 7 `models/record_z_c1_seed7.joblib`, SHA-256 `3d90232f990982bc5a29d7dede45eddadf0c6093e14713a1cb9abd91435ac750`; seed 123 `models/record_z_c1_seed123.joblib`, SHA-256 `783096b78e10af04d5913038bf3b073c0519c7bb21f68f67bf108060464cf917`; seed-42 repeat `models/record_z_c1_seed42_repeat.joblib`, SHA-256 `ff0c3ff905ee514c5983fa7edcb2860920182a540d2ac701c6a76cf8a7481831`. `champion.json` now points at `record_z_c1`.

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
