from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

MAX_ROWS = 100_000


def _finite_probability(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= float(value) <= 1
    ):
        raise ValueError("score must be a finite number in [0,1]")
    return float(value)


def _label(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("label must be the integer 0 or 1")
    return value


def load_records(path: Path) -> list[tuple[int, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_ROWS:
        raise ValueError(f"input must be a non-empty list with <= {MAX_ROWS} rows")
    records: list[tuple[int, float]] = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("each row must be an object")
        records.append((_label(row.get("label")), _finite_probability(row.get("score"))))
    return records


def confusion(records: list[tuple[int, float]], threshold: float) -> dict[str, int]:
    threshold = _finite_probability(threshold)
    true_positive = false_positive = true_negative = false_negative = 0
    for raw_label, raw_score in records:
        label = _label(raw_label)
        score = _finite_probability(raw_score)
        prediction = int(score >= threshold)
        if label and prediction:
            true_positive += 1
        elif not label and prediction:
            false_positive += 1
        elif not label and not prediction:
            true_negative += 1
        else:
            false_negative += 1
    return {
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def metrics(confusion_matrix: dict[str, int]) -> dict[str, float]:
    true_positive, false_positive, true_negative, false_negative = (
        confusion_matrix[key] for key in ("tp", "fp", "tn", "fn")
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / total if total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def sweep(records: list[tuple[int, float]], step: float = 0.05) -> list[dict[str, float | int]]:
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(step)
        or not 0 < step <= 1
    ):
        raise ValueError("step must be a finite number in (0,1]")
    rows: list[dict[str, float | int]] = []
    current = 0.0
    while current < 1.0:
        threshold = round(current, 10)
        matrix = confusion(records, threshold)
        rows.append({"threshold": threshold, **matrix, **metrics(matrix)})
        current += float(step)
    if not rows or rows[-1]["threshold"] != 1.0:
        matrix = confusion(records, 1.0)
        rows.append({"threshold": 1.0, **matrix, **metrics(matrix)})
    return rows


def best_by_f1(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("no sweep rows")
    return max(
        rows,
        key=lambda row: (
            row["f1"],
            row["recall"],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )


def render_html(rows: list[dict[str, float | int]], output: Path) -> None:
    best = best_by_f1(rows)
    width, height, padding = 760, 220, 34
    points = []
    for row in rows:
        x = padding + row["threshold"] * (width - 2 * padding)
        y = height - padding - row["f1"] * (height - 2 * padding)
        points.append(f"{x:.1f},{y:.1f}")
    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(f'{row[key]:.3f}' if isinstance(row[key], float) else str(row[key]))}</td>"
            for key in ("threshold", "precision", "recall", "f1", "accuracy")
        )
        + "</tr>"
        for row in rows
    )
    document = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>AI Threshold Playground</title>
<style>body{{font:16px system-ui;margin:32px;max-width:960px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ccc;text-align:right}}.card{{border:1px solid #ccc;border-radius:12px;padding:16px;margin:16px 0}}svg{{max-width:100%;height:auto}}</style>
<body>
<h1>AI Threshold Playground</h1>
<p>Local evaluation report. No model inference or network calls are performed.</p>
<div class="card"><b>Best F1 threshold:</b> {best['threshold']:.2f} &nbsp; <b>F1:</b> {best['f1']:.3f}</div>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="F1 across thresholds"><rect x="0" y="0" width="100%" height="100%" fill="white" stroke="#ccc"/><polyline fill="none" stroke="currentColor" stroke-width="3" points="{' '.join(points)}"/></svg>
<table><thead><tr><th>threshold</th><th>precision</th><th>recall</th><th>f1</th><th>accuracy</th></tr></thead><tbody>{table}</tbody></table>
</body>
</html>"""
    output.write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate recorded binary classifier scores without network access."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--html", type=Path)
    arguments = parser.parse_args(argv)
    records = load_records(arguments.input)
    rows = sweep(records, arguments.step)
    best = best_by_f1(rows)
    print(json.dumps({"rows": len(records), "best": best}, ensure_ascii=False, indent=2))
    if arguments.html:
        render_html(rows, arguments.html)


if __name__ == "__main__":
    main()
