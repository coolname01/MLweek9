# Sleep-EDF: sleep stage recognition (sleep-edf-neuroscience-2026)

Model: 30 s x 3 channels (EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal) -> one of 5 stages (0 W, 1 N1, 2 N2, 3 N3, 4 REM). Metric: macro-F1.
Edition `full_psg_v1` (updated 28 Oct 2026): full records on the original PSG grid, ids start with `full_`. Old caches, predictions and numbers from truncated versions do not apply.
Source: Sleep-EDF Expanded, Sleep Cassette (Bob Kemp, PhysioNet), ODC-BY 1.0. Labels: Rechtschaffen-Kales, stages 3+4 merged; Movement and "?" excluded. Teaching model, not a clinical system.

## Dataset (as stated on the competition page; verify locally)
| Part | Records | Epochs | Use |
|---|---|---|---|
| Train | 112 | 302,653 | fit on open labels |
| Val | 16 | 43,206 | check on new subjects |
| Test | 24 | 66,222 | Kaggle predictions, labels hidden |
Total: 152 records, 412,081 epochs. Subjects do not overlap across parts. Private leaderboard: 6 subjects.

## Data format
- NPZ per record: `x` (N,3000,3) float32 uV, `ids`, `epoch_index`, `fs`=100, `channel_names`.
- `labels.csv`: id,label,split,record_id,epoch_index,subject_id. `sample_submission.csv`: id,label (keep ids and row order).
- Join by `id`. Time of an epoch = 30 * epoch_index s. A jump in epoch_index > 1 is a gap: break context there and at file boundaries.
- Long wake periods are kept, so W is over-represented. Compute class distribution from the delivered data.

## Layout
```
data/ models/        local only (not in git)
notebooks/           checkpoint, improved, final
src/                 data.py features.py models.py train.py infer.py
configs/config.yaml  seed, paths
experiments/results.csv
README.md AGENTS.md docs/STATE.md PROMPTS.md
```

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas scipy scikit-learn matplotlib joblib jupyter lightgbm pyyaml
```

## Git / GitHub
```
git init
git add .
git commit -m "chore: project skeleton"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
Commit after every experiment; tag finals (`final-1`, `final-2`).

## Rules summary
- Only competition data; no external datasets, original hypnograms, or matching test signals to original recordings.
- Test: predictions only. No training, pseudo-labels, test-time adaptation, BatchNorm updates, or test-derived stats/thresholds. Predictions for one record must not depend on other test records.
- Allowed: fixed per-record transforms (filters, features, normalization by the record's own mean/std) and context of neighboring epochs of the same record.
- Pretrained weights: disclose name, source, exact checkpoint/hash, pretraining corpora (incl. Sleep-EDF overlap; "unknown" if unknown), frozen vs fine-tuned, input conversion (uV->V = 1e-6, axes (N,C,T)). Not guaranteed to fit 3 channels (BENDR, LaBraM, CBraMod, BIOT).
- Final Kaggle run: clean kernel, no internet, CPU or T4x2, <= 2 h, loads a saved checkpoint, deterministic. 3 submissions/day, up to 2 finals.
- Disclose AI help (tool, purpose, verification).
