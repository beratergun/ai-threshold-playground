# AI Threshold Playground

AI Threshold Playground is a small local toolkit for exploring what happens when the decision threshold of a binary classifier changes.

It starts from recorded labels and probability-like scores. There is no model inference in the project; the focus is the evaluation step after scores already exist.

## Why this project

A single accuracy value can hide the trade-off between false positives and false negatives. I wanted a compact implementation where changing the threshold makes that trade-off visible immediately.

The project is useful for experimenting with how precision, recall and F1 move even when the underlying model scores stay exactly the same.

## What it calculates

For each threshold the tool reports:

- True positives
- False positives
- True negatives
- False negatives
- Precision
- Recall
- F1
- Accuracy

It can sweep thresholds across `[0, 1]`, select the best row by a deterministic F1-based rule and generate a self-contained HTML report with an SVG curve and metric table.

## Run locally

```bash
python threshold_lab.py examples/sample.json
```

Generate an HTML report:

```bash
python threshold_lab.py examples/sample.json --html report.html
```

A custom sweep step can also be supplied with `--step`.

## Input validation

Labels must be integer `0` or `1`. Scores and thresholds must be finite values between `0` and `1`, and the input dataset is bounded to a defined maximum row count.

These checks keep malformed recorded data from silently influencing the result.

## Limitations

The “best” threshold in this repository is best only according to the implemented ranking rule. Real applications may care more about recall, precision, expected cost, class imbalance or domain-specific risk.

The tool also does not train a model or claim that F1 is the correct objective for every problem.

## Possible next steps

I would like to add explicit cost-sensitive thresholding, PR/ROC summaries and side-by-side comparison of several recorded score sets.

## License

See [LICENSE](LICENSE).
