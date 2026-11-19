# Run the clinic decision tree in Cursor

This folder rebuilds the v1 clinic no-show analysis: a depth-3 decision tree on 280 training appointments, scored on a fixed holdout of 120.

| File | What it is |
|---|---|
| `clinic_appointments.csv` | 400 teaching appointments. Not real patient records. |
| `decision_tree_analysis.py` | Fits the tree, checks the published numbers, writes the charts and `analysis_results.md`. |
| `analysis_results.md` | The written results. Regenerated when you run the script. |
| `requirements.txt` | Pinned packages used to draw the figures. |
| `outputs/` | The three charts, created by the script. |

Headline holdout numbers the script checks before it writes anything:

- Current policy: 77.5% accuracy, 0% no-show precision, 0% no-show recall, **$27.00** average loss
- Decision tree v1: 88.3% accuracy, 74.1% no-show precision, 74.1% no-show recall, **$10.50** average loss

## Setup

Open a terminal in Cursor (**Terminal → New Terminal**) and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead of `source`.

The pins are `numpy==2.5.5`, `pandas==3.1.1`, `scikit-learn==1.9.1`, and `matplotlib==3.11.2`. The gini labels on the tree use scikit-learn's default 3-decimal rounding (`0.353`, `0.43`, and so on).

## Run

From this folder:

```bash
python decision_tree_analysis.py
```

The script always reads `clinic_appointments.csv` next to itself, so the working directory can be somewhere else.

You should see:

```text
Training appointments: 280
Holdout appointments: 120
Tree depth: 3   leaves: 5

Holdout accuracy:        Current policy 77.5% | Decision tree v1 88.3%
No-show precision:       Current policy 0.0% | Decision tree v1 74.1%
No-show recall:          Current policy 0.0% | Decision tree v1 74.1%
Average loss per visit:  Current policy $27.00 | Decision tree v1 $10.50
Holdout total loss:      Current policy $3,240.00 | Decision tree v1 $1,260.00

Wrote outputs/decision_tree_v1.png
Wrote outputs/test_set_cost.png
Wrote outputs/holdout_performance.png
Wrote analysis_results.md
```

Then open, from the Cursor explorer:

- `outputs/decision_tree_v1.png`
- `outputs/holdout_performance.png`
- `outputs/test_set_cost.png`
- `analysis_results.md`

If a check fails, the script exits before replacing those files. The usual cause is an edit to the two model columns, the depth, or the `appointment_id` split.

## What v1 actually fits

Inputs are `prior_no_shows` and `sms_reminder`. The target is `outcome` (`checked-in` or `no-show`).

`age`, `lead_time_days`, and `weekday` are on the file so you can try them. Adding any of them to `FEATURES` in `decision_tree_analysis.py` fits a different tree, and the node-count check will stop the run. That is the point of the check: the published figure is this two-feature tree.

The holdout is fixed: `appointment_id` 1–280 train, 281–400 test. `train_test_split` would reshuffle those counts and would not redraw the v1 figure.

Loss used for the cost chart, charged only on a wrong call:

- missed no-show (empty slot): $120
- false alarm (extra outreach or an overbook of someone who arrives): $60

`analysis_results.md` has the leaf counts, the confusion matrix, and that arithmetic.
