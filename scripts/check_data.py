"""Integrity checks for the local full_psg_v1 checkout.

Reads one NPZ at a time. Official record and epoch counts are fixed;
a mismatch is reported and is not rewritten to match the files.
``team_info.csv`` is never opened.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    EXPECTED_EPOCHS,
    EXPECTED_RECORDS,
    LABEL_COLUMNS,
    SPLITS,
    SUBMISSION_COLUMNS,
    Config,
    file_sha256,
    gap_lengths,
    load_config,
    load_record,
    manifest_local_path,
    read_labels,
    read_submission,
    record_format_issues,
    record_paths,
)


@dataclass
class RecordSummary:
    """Gap facts kept after the signal of a record is released."""

    split: str
    record_id: str
    n_epochs: int
    gap_lengths: list[int]
    issues: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    """Collected outcomes. Counts stay equal to what was measured."""

    summaries: list[RecordSummary] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    manifest_matched: int = 0
    manifest_skipped: int = 0
    manifest_compared: int = 0
    subject_counts: dict[str, int] = field(default_factory=dict)
    subject_overlap: int = 0
    elapsed_s: float = 0.0


def _add(result: CheckResult, message: str) -> None:
    result.issues.append(message)


def _missing_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [name for name in columns if name not in frame.columns]


def _blank_mask(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for name in columns:
        column = frame[name]
        mask = mask | column.isna() | column.str.strip().eq("")
    return mask


def _scan_records(config: Config, result: CheckResult) -> dict[str, tuple[str, str, int]]:
    """Load each NPZ once and index ``id`` -> (split, record_id, epoch_index)."""
    located: dict[str, tuple[str, str, int]] = {}
    for split in SPLITS:
        paths = record_paths(config.signals_dir, split)
        if not paths and not (config.signals_dir / split).is_dir():
            _add(result, f"missing directory {split}/")
        for path in paths:
            try:
                record = load_record(path, split)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                _add(result, f"{split}/{path.name}: unreadable ({type(exc).__name__}: {exc})")
                result.summaries.append(
                    RecordSummary(split, path.stem, 0, [], [f"unreadable: {exc}"])
                )
                continue
            issues = record_format_issues(record, config)
            gaps = gap_lengths(record.epoch_index)
            limit = min(len(record.ids), len(record.epoch_index))
            for offset in range(limit):
                epoch_id = record.ids[offset]
                previous = located.get(epoch_id)
                if previous is not None:
                    _add(
                        result,
                        f"id {epoch_id} is in {previous[0]}/{previous[1]} "
                        f"and {split}/{record.record_id}",
                    )
                else:
                    located[epoch_id] = (split, record.record_id, record.epoch_index[offset])
            result.summaries.append(
                RecordSummary(
                    split=split,
                    record_id=record.record_id,
                    n_epochs=record.n_epochs,
                    gap_lengths=gaps,
                    issues=issues,
                )
            )
            for issue in issues:
                _add(result, f"{split}/{record.record_id}: {issue}")
            print(
                f"{split}/{record.record_id}: epochs={record.n_epochs} gaps={len(gaps)}",
                file=sys.stderr,
                flush=True,
            )
    return located


def _check_counts(result: CheckResult) -> None:
    for split in SPLITS:
        rows = [row for row in result.summaries if row.split == split]
        n_records = len(rows)
        n_epochs = sum(row.n_epochs for row in rows)
        if n_records != EXPECTED_RECORDS[split] or n_epochs != EXPECTED_EPOCHS[split]:
            _add(
                result,
                f"{split}: records {n_records} (expected {EXPECTED_RECORDS[split]}), "
                f"epochs {n_epochs} (expected {EXPECTED_EPOCHS[split]})",
            )


def _check_labels(
    config: Config,
    result: CheckResult,
    located: dict[str, tuple[str, str, int]],
) -> None:
    path = config.tables_dir / "labels.csv"
    if not path.is_file():
        _add(result, "labels.csv is missing")
        return
    labels = read_labels(path)
    missing = _missing_columns(labels, LABEL_COLUMNS)
    if missing:
        _add(result, f"labels.csv missing columns {missing}")
        return
    blank = int(_blank_mask(labels, list(LABEL_COLUMNS)).sum())
    if blank:
        _add(result, f"labels.csv has {blank} rows with an empty required field")
    duplicated = int(labels["id"].duplicated().sum())
    if duplicated:
        _add(result, f"labels.csv has {duplicated} duplicate ids")

    usable = labels.loc[~_blank_mask(labels, ["id", "split", "record_id", "epoch_index"])]
    mismatches = 0
    preview: list[str] = []

    def note(text: str) -> None:
        nonlocal mismatches
        mismatches += 1
        if len(preview) < 8:
            preview.append(text)

    for row in usable.itertuples(index=False):
        epoch_id = str(row.id)
        split = str(row.split)
        record_id = str(row.record_id)
        if split not in ("train", "val"):
            note(f"{epoch_id} split={split}")
            continue
        found = located.get(epoch_id)
        if found is None:
            note(f"{epoch_id} absent from NPZ")
            continue
        found_split, found_record, found_epoch = found
        if found_split != split or found_record != record_id:
            note(
                f"{epoch_id} table {split}/{record_id} "
                f"file {found_split}/{found_record}"
            )
            continue
        try:
            table_epoch = int(str(row.epoch_index))
        except ValueError:
            note(f"{epoch_id} epoch_index is not an integer")
            continue
        if table_epoch != found_epoch:
            note(f"{epoch_id} epoch_index table {table_epoch} file {found_epoch}")
    if mismatches:
        _add(
            result,
            f"labels.csv join mismatches: {mismatches} ({'; '.join(preview)})",
        )

    for split in ("train", "val"):
        file_ids = {
            epoch_id
            for epoch_id, (file_split, _, _) in located.items()
            if file_split == split
        }
        table_ids = set(usable.loc[usable["split"] == split, "id"].astype(str))
        only_files = len(file_ids - table_ids)
        only_table = len(table_ids - file_ids)
        if only_files or only_table:
            _add(
                result,
                f"{split} id coverage: {only_files} ids only in NPZ, "
                f"{only_table} ids only in labels.csv",
            )

    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        part = labels.loc[labels["split"] == split, "label"]
        numeric = pd.to_numeric(part, errors="coerce")
        counter: Counter[str] = Counter()
        for value in numeric.tolist():
            if pd.isna(value) or int(value) != value or int(value) not in range(len(config.classes)):
                counter["invalid"] += 1
            else:
                counter[config.classes[int(value)]] += 1
        counts[split] = {name: int(counter.get(name, 0)) for name in config.classes}
        if counter.get("invalid", 0):
            counts[split]["invalid"] = int(counter["invalid"])
            _add(result, f"{split} has {counter['invalid']} labels outside 0..4")
    result.class_counts = counts

    def subjects(split: str) -> set[str]:
        column = labels.loc[labels["split"] == split, "subject_id"].dropna()
        return {str(value).strip() for value in column if str(value).strip()}

    train_subjects = subjects("train")
    val_subjects = subjects("val")
    overlap = sorted(train_subjects & val_subjects)
    result.subject_counts = {"train": len(train_subjects), "val": len(val_subjects)}
    result.subject_overlap = len(overlap)
    if overlap:
        _add(
            result,
            f"subject_id overlap between train and val: {len(overlap)} "
            f"({', '.join(overlap[:8])})",
        )


def _check_submission(
    config: Config,
    result: CheckResult,
    located: dict[str, tuple[str, str, int]],
) -> None:
    path = config.tables_dir / "sample_submission.csv"
    if not path.is_file():
        _add(result, "sample_submission.csv is missing")
        return
    submission = read_submission(path)
    missing = _missing_columns(submission, SUBMISSION_COLUMNS)
    if missing:
        _add(result, f"sample_submission.csv missing columns {missing}")
        return
    blank = int(_blank_mask(submission, ["id"]).sum())
    if blank:
        _add(result, f"sample_submission.csv has {blank} empty ids")
    duplicated = int(submission["id"].duplicated().sum())
    if duplicated:
        _add(result, f"sample_submission.csv has {duplicated} duplicate ids")
    submission_ids = set(submission["id"].dropna().astype(str))
    test_ids = {
        epoch_id
        for epoch_id, (split, _, _) in located.items()
        if split == "test"
    }
    missing_in_test = sorted(submission_ids - test_ids)
    missing_in_template = sorted(test_ids - submission_ids)
    if missing_in_test:
        preview = ", ".join(missing_in_test[:8])
        _add(
            result,
            f"sample_submission ids absent from data/test: {len(missing_in_test)} ({preview})",
        )
    if missing_in_template:
        preview = ", ".join(missing_in_template[:8])
        _add(
            result,
            f"data/test ids absent from sample_submission: {len(missing_in_template)} ({preview})",
        )


def _check_manifest(config: Config, result: CheckResult) -> None:
    path = config.tables_dir / "MANIFEST.json"
    if not path.is_file():
        _add(result, "MANIFEST.json is missing")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list):
        _add(result, "MANIFEST.json has no files list")
        return
    has_size = any(isinstance(item, dict) and "bytes" in item for item in files)
    has_hash = any(isinstance(item, dict) and "sha256" in item for item in files)
    if not has_size and not has_hash:
        _add(result, "MANIFEST.json files have neither bytes nor sha256")
        return
    seen_local: set[Path] = set()
    for item in files:
        if not isinstance(item, dict) or "path" not in item:
            _add(result, "MANIFEST.json contains an entry without path")
            continue
        rel = str(item["path"])
        local = manifest_local_path(rel, config)
        if local is None:
            result.manifest_skipped += 1
            continue
        result.manifest_compared += 1
        seen_local.add(local.resolve())
        if not local.is_file():
            _add(
                result,
                "manifest file missing locally: "
                f"{rel} (looked at {local.relative_to(config.root).as_posix()}, "
                f"manifest bytes {item.get('bytes')})",
            )
            continue
        if has_size and "bytes" in item and local.stat().st_size != int(item["bytes"]):
            _add(
                result,
                f"size mismatch {rel}: local {local.stat().st_size} "
                f"manifest {item['bytes']}",
            )
            continue
        if has_hash and "sha256" in item:
            digest = file_sha256(local)
            if digest != str(item["sha256"]):
                _add(result, f"sha256 mismatch {rel}")
                continue
        result.manifest_matched += 1

    for split in SPLITS:
        for record_path in record_paths(config.signals_dir, split):
            if record_path.resolve() not in seen_local:
                _add(result, f"NPZ not listed in manifest: {split}/{record_path.name}")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _count_rows(result: CheckResult) -> list[list[str]]:
    rows: list[list[str]] = []
    for split in SPLITS:
        part = [row for row in result.summaries if row.split == split]
        n_records = len(part)
        n_epochs = sum(row.n_epochs for row in part)
        rows.append(
            [
                split,
                str(n_records),
                str(EXPECTED_RECORDS[split]),
                str(n_epochs),
                str(EXPECTED_EPOCHS[split]),
                "ok"
                if n_records == EXPECTED_RECORDS[split]
                and n_epochs == EXPECTED_EPOCHS[split]
                else "mismatch",
            ]
        )
    return rows


def _class_rows(config: Config, result: CheckResult) -> list[list[str]]:
    rows: list[list[str]] = []
    for split in ("train", "val"):
        counts = result.class_counts.get(split, {})
        total = sum(counts.get(name, 0) for name in config.classes)
        for index, name in enumerate(config.classes):
            count = counts.get(name, 0)
            share = 0.0 if total == 0 else 100.0 * count / total
            rows.append([split, name, str(index), str(count), f"{share:.2f}"])
        if counts.get("invalid"):
            rows.append([split, "invalid", "", str(counts["invalid"]), ""])
    return rows


def _gap_rows(result: CheckResult) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in result.summaries:
        lengths = " ".join(str(length) for length in row.gap_lengths)
        rows.append(
            [
                row.split,
                row.record_id,
                str(row.n_epochs),
                str(len(row.gap_lengths)),
                str(sum(row.gap_lengths)),
                lengths,
            ]
        )
    return rows


def render_report(config: Config, result: CheckResult) -> str:
    """Build the markdown report. It does not include roster fields."""
    passed = not result.issues
    versions = (
        f"Python {platform.python_version()}, "
        f"numpy {np.__version__}, pandas {pd.__version__}, pyyaml {yaml.__version__}"
    )
    lines = [
        "# Data checks",
        "",
        "Updated: 2026-11-02. Status: **"
        + ("passed" if passed else f"{len(result.issues)} discrepancy listed")
        + "**.",
        "",
        "One NPZ at a time (`np.load`). `team_info.csv` was not opened. "
        "Official record and epoch counts were not changed to fit the files.",
        "",
        f"Runtime: {result.elapsed_s:.1f} s. {versions}. {platform.platform()}.",
        f"Seed in config: {config.seed}. Signal contract: fs={config.fs}, "
        f"epoch_length={config.epoch_length}, channels={list(config.channels)}.",
        "",
        "## Counts",
        "",
        _markdown_table(
            ["split", "records", "expected records", "epochs", "expected epochs", "status"],
            _count_rows(result),
        ),
        "",
        "## Subjects",
        "",
        f"Unique `subject_id`: train {result.subject_counts.get('train', 0)}, "
        f"val {result.subject_counts.get('val', 0)}, "
        f"overlap {result.subject_overlap}. Identifiers are not listed.",
        "",
        "## Class distribution",
        "",
        "Shares are within the split, in percent. Labels come from `labels.csv`.",
        "",
        _markdown_table(
            ["split", "class", "label", "epochs", "share %"],
            _class_rows(config, result),
        ),
        "",
        "## epoch_index gaps",
        "",
        "A gap is a forward step greater than 1. Length is the number of missing "
        "30-second slots (`step - 1`). Lengths are listed in file order.",
        "",
        _markdown_table(
            ["split", "record_id", "epochs", "n_gaps", "missing slots", "gap lengths"],
            _gap_rows(result),
        ),
        "",
        "## Manifest",
        "",
        f"Compared: {result.manifest_compared}. "
        f"Bytes and SHA-256 matched: {result.manifest_matched}. "
        f"Skipped unread `team_info.csv`: {result.manifest_skipped}.",
        "",
        "## Discrepancies",
        "",
    ]
    if result.issues:
        lines.extend(f"- {issue}" for issue in result.issues)
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def render_chat(config: Config, result: CheckResult) -> str:
    """Short table for the console."""
    count_rows = _count_rows(result)
    class_bits = []
    for split in ("train", "val"):
        counts = result.class_counts.get(split, {})
        text = ", ".join(
            f"{name} {counts.get(name, 0)}" for name in config.classes
        )
        class_bits.append(f"{split}: {text}")
    n_gaps = sum(len(row.gap_lengths) for row in result.summaries)
    missing = sum(sum(row.gap_lengths) for row in result.summaries)
    rows = [
        ["records / epochs", "; ".join(
            f"{row[0]} {row[1]}/{row[3]} ({row[5]})" for row in count_rows
        )],
        ["classes", "; ".join(class_bits) if class_bits else "not computed"],
        ["gaps", f"{n_gaps} gaps, {missing} missing slots"],
        [
            "manifest",
            f"matched {result.manifest_matched}/{result.manifest_compared}, "
            f"team_info skipped {result.manifest_skipped}",
        ],
        ["result", "passed" if not result.issues else f"{len(result.issues)} discrepancy listed"],
    ]
    return _markdown_table(["check", "value"], rows)


def main() -> int:
    """Run the checks and write ``docs/data_summary.md``."""
    started = time.perf_counter()
    config = load_config(ROOT / "configs" / "config.yaml")
    np.random.seed(config.seed)
    result = CheckResult()
    located = _scan_records(config, result)
    _check_counts(result)
    _check_labels(config, result, located)
    _check_submission(config, result, located)
    _check_manifest(config, result)
    result.elapsed_s = time.perf_counter() - started
    report_path = ROOT / "docs" / "data_summary.md"
    report_path.write_text(render_report(config, result), encoding="utf-8")
    print(render_chat(config, result))
    print(f"wrote {report_path.relative_to(ROOT)}")
    return 0 if not result.issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
