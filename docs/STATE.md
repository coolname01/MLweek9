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
- [ ] Baseline: Welch PSD band features + amplitude stats + classical classifier, val macro-F1, per-class F1, time
- [ ] Hypothesis 1: neighboring-epoch context within a record
- [ ] Final checkpoint + SHA-256, offline Kaggle notebook (<= 2 h)

## Results
See experiments/results.csv (empty).

## Layout (actual, 2026-11-02)
- Signals: `data/train` 112 npz, `data/val` 16, `data/test` 24.
- `labels.csv`, `sample_submission.csv`, `team_info.csv`, `MANIFEST.json` and competition `*.md` are in `archive/`, not next to the splits.
- State file: `docs/STATE.md`.

## Next step
Baseline: Welch PSD band features + amplitude stats + a classical classifier. Select on train/val only.

Checks ran on Python 3.14.8 (3.12 is not installed here), numpy 2.5.4, pandas 3.1.0, pyyaml 6.0.3, 58 s.

## Disclosures
- AI: Perplexity (planning, prompts, project files); Cursor (code) - describe parts and verification.
- External code / pretrained weights: none yet.
