# AGENTS.md - rules for Cursor / AI agents

## Task
Sleep staging, 5 classes, macro-F1, `full_psg_v1` (112 train / 16 val / 24 test records). Read README.md and docs/STATE.md first; update docs/STATE.md after each task.

## Hard constraints
1. Use only provided competition files. No external datasets, no original hypnograms, no matching test signals to original recordings or recovering their names, no older truncated versions.
2. Never fit anything on test signals (models, scalers, PCA, pseudo-labels, TTA, BatchNorm stats). Do not tune thresholds, ensembles or parameters on the test distribution. A test record's prediction must not depend on other test records.
3. Allowed: fixed per-record deterministic transforms (filters, features, normalization by the record's own mean/std), context from the same record only.
4. Select methods on train/val, then freeze. Final model may be retrained on train+val; keep the earlier validation results.
5. Pretrained weights only with full disclosure and checks of channels, fs, input length, units (uV vs V) and axis order.

## Data handling
- Join by `id`, never by row order. Assert: unique ids, no NaN/inf, shape (N,3000,3), float32, channel order, signal count == label count.
- Validate by subject (`subject_id`) or the given train/val split; never split epochs randomly.
- Context windows are per record and break at epoch_index gaps (>1) and file boundaries.
- Report per-class F1 and a confusion matrix, not only accuracy. W dominates.
- Memory: ~412k epochs x 3000 x 3 float32 is ~15 GB. Use per-record loading / memmap and cache features, not raw signals.

## Engineering
- Python 3.12, type hints, NumPy-style docstrings, minimal comments. Parameters in `configs/config.yaml`.
- Fixed seeds, deterministic inference, record library versions, hardware, training time.
- Log every experiment in `experiments/results.csv`: id, hypothesis, features, model, val macro-F1, per-class F1, time, seed, commit.
- Save checkpoints in `models/` with SHA-256. Never commit data, weights, secrets.
- Final notebook: offline, <= 2 h, writes `submission.csv` (id,label; sample order; labels 0-4).

## Git
Small commits (`feat:`, `exp:`, `fix:`). No force-push or history rewrite. Ask before destructive commands.

## Communication
Reply to the user in Russian; code, identifiers, commits in English.
