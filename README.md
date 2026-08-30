# AI Threshold Playground

AI Threshold Playground is a local evaluation tool for understanding how a recorded binary
classifier score set changes as the decision threshold moves. It does not call a model,
provider, or external API.

## Problem

A single accuracy value can hide precision/recall trade-offs, and threshold selection can
be difficult to review when the underlying calculations are opaque.

## Why this project exists

The project provides a small, deterministic reference implementation that makes every
confusion-matrix count and derived metric inspectable from recorded labels and scores.

## Features

- Bounded, non-empty JSON input containing integer 'label' and finite 'score' values
- Confusion matrix, precision, recall, F1, and accuracy
- Deterministic threshold sweep that always includes 0.0 and 1.0
- Best-F1 selection with explicit deterministic tie-breaking
- Self-contained responsive HTML/SVG report
- No model inference, telemetry, or network access

## Project structure

- 'threshold_lab.py': parsing, validation, metrics, sweep, CLI, and HTML renderer
- 'examples/sample.json': synthetic recorded-score fixture
- 'tests/test_threshold_lab.py': deterministic unit and boundary tests
- '.github/workflows/tests.yml': clean-checkout Python test job

## Setup

Python 3.12 is used in CI. The project has no third-party runtime dependencies.

## Usage

~~~bash
python threshold_lab.py examples/sample.json --step 0.05 --html report.html
~~~

'report.html' is generated output and is intentionally ignored by Git.

## Tests

~~~bash
python -m unittest discover -s tests -v
~~~

Tests cover metrics, deterministic boundaries and tie-breaking, invalid probabilities and
labels, row limits, invalid sweep steps, and the self-contained report.

## Engineering decisions

- The tool accepts recorded scores instead of performing inference, keeping evaluation
  reproducible and provider-independent.
- Input rows, labels, probabilities, and sweep steps are validated before evaluation.
- The report embeds its CSS and SVG and does not load remote assets.

## Limitations

- Binary classification only
- No ROC/AUC or calibration metrics
- No model inference or live dataset connector
- The visualization is intentionally lightweight

## Possible improvements

- Add ROC/AUC and calibration views.
- Add multiclass evaluation.
- Add CSV import with an explicit validated schema.
- Add an optional machine-readable report export.

## Security and privacy

Input remains local. The program has no network client, telemetry, model call, or storage
service. Generated reports contain the calculated aggregate rows and should be reviewed
before sharing when the source scores are sensitive.

## License

Licensed under the MIT License. See 'LICENSE'.
