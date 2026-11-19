"""Fit clinic no-show decision tree v1 and write the holdout charts.

The training rows are appointment_id 1-280 and the holdout rows are
appointment_id 281-400. That split is fixed so the figure matches the
published v1 tree. A fresh random split will not.

v1 uses two inputs: prior_no_shows and sms_reminder.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "clinic_appointments.csv"
OUTPUT_DIR = ROOT / "outputs"
REPORT_PATH = ROOT / "analysis_results.md"

FEATURES = ["prior_no_shows", "sms_reminder"]
TARGET = "outcome"
TRAIN_ID_MAX = 280
HOLDOUT_SIZE = 120

# Planning loss, in dollars, charged only when the call is wrong.
# A missed no-show is an empty slot. A false alarm is extra outreach,
# or an overbook, for someone who checks in.
MISSED_NO_SHOW_USD = 120
FALSE_ALARM_USD = 60

CHECKED_IN = "checked-in"
NO_SHOW = "no-show"
CURRENT_POLICY = "Current policy"
TREE_NAME = "Decision tree v1"
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"


def load_appointments(path=DATA_PATH):
    frame = pd.read_csv(path)
    expected = [
        "appointment_id",
        "age",
        "lead_time_days",
        "weekday",
        "prior_no_shows",
        "sms_reminder",
        "outcome",
    ]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns: {missing}")

    if frame["appointment_id"].nunique() != len(frame):
        raise SystemExit("appointment_id values must be unique")
    if set(frame["outcome"]) != {CHECKED_IN, NO_SHOW}:
        raise SystemExit("outcome must be checked-in or no-show")
    if not set(frame["sms_reminder"]).issubset({0, 1}):
        raise SystemExit("sms_reminder must be 0 or 1")

    train = frame.loc[frame["appointment_id"] <= TRAIN_ID_MAX].copy()
    holdout = frame.loc[frame["appointment_id"] > TRAIN_ID_MAX].copy()
    if len(train) != TRAIN_ID_MAX or len(holdout) != HOLDOUT_SIZE:
        raise SystemExit(
            f"Expected {TRAIN_ID_MAX} training rows and {HOLDOUT_SIZE} holdout "
            f"rows, found {len(train)} and {len(holdout)}."
        )
    return train, holdout


def fit_tree(train):
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(train[FEATURES], train[TARGET])
    return model


def class_counts(tree, node_id):
    """Return [checked-in, no-show] counts for one node."""
    raw = np.asarray(tree.value[node_id][0], dtype=float)
    scaled = raw * tree.weighted_n_node_samples[node_id]
    return np.rint(scaled).astype(int)


def assert_published_tree(model):
    """Stop if the fitted tree would not redraw the v1 figure."""
    tree = model.tree_
    if list(model.classes_) != [CHECKED_IN, NO_SHOW]:
        raise SystemExit(f"Unexpected class order: {list(model.classes_)}")
    if model.get_depth() != 3 or model.get_n_leaves() != 5:
        raise SystemExit("v1 is a depth-3 tree with 5 leaves")

    # Node order is sklearn's left-first layout.
    # feature -2 is a leaf. Counts are [checked-in, no-show].
    expected = [
        # node, feature, threshold, samples, counts, gini label
        (0, 0, 2.5, 280, (216, 64), "0.353"),
        (1, 1, 0.5, 263, (214, 49), "0.303"),
        (2, 0, 0.5, 108, (69, 39), "0.461"),
        (3, -2, None, 56, (46, 10), "0.293"),
        (4, -2, None, 52, (23, 29), "0.493"),
        (5, 0, 1.5, 155, (145, 10), "0.121"),
        (6, -2, None, 139, (134, 5), "0.069"),
        (7, -2, None, 16, (11, 5), "0.43"),
        (8, -2, None, 17, (2, 15), "0.208"),
    ]
    if tree.node_count != len(expected):
        raise SystemExit(f"Expected 9 nodes, found {tree.node_count}")

    for node, feature, threshold, samples, counts, gini_label in expected:
        if int(tree.feature[node]) != feature:
            raise SystemExit(f"Node {node} split on feature {tree.feature[node]}")
        if threshold is not None and abs(float(tree.threshold[node]) - threshold) > 1e-6:
            raise SystemExit(f"Node {node} threshold is {tree.threshold[node]}")
        if int(tree.n_node_samples[node]) != samples:
            raise SystemExit(f"Node {node} has {tree.n_node_samples[node]} samples")
        got = tuple(int(value) for value in class_counts(tree, node))
        if got != counts:
            raise SystemExit(f"Node {node} counts are {got}, expected {counts}")
        label = str(round(float(tree.impurity[node]), 3))
        if label != gini_label:
            raise SystemExit(f"Node {node} gini prints as {label}, expected {gini_label}")

    children = {
        0: (1, 8),
        1: (2, 5),
        2: (3, 4),
        5: (6, 7),
    }
    for node, (left, right) in children.items():
        if int(tree.children_left[node]) != left or int(tree.children_right[node]) != right:
            raise SystemExit(f"Node {node} children do not match the v1 figure")


def loss_counts(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    missed = int(np.sum((y_true == NO_SHOW) & (y_pred == CHECKED_IN)))
    false_alarm = int(np.sum((y_true == CHECKED_IN) & (y_pred == NO_SHOW)))
    total = missed * MISSED_NO_SHOW_USD + false_alarm * FALSE_ALARM_USD
    return missed, false_alarm, total


def binary_rates(y_true, y_pred):
    """Accuracy, no-show precision, and no-show recall as percents."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = 100.0 * accuracy_score(y_true, y_pred)
    precision = 100.0 * precision_score(
        y_true, y_pred, pos_label=NO_SHOW, zero_division=0
    )
    recall = 100.0 * recall_score(
        y_true, y_pred, pos_label=NO_SHOW, zero_division=0
    )
    return accuracy, precision, recall


def assert_published_holdout(y_true, current_pred, tree_pred):
    missed_c, false_c, total_c = loss_counts(y_true, current_pred)
    missed_t, false_t, total_t = loss_counts(y_true, tree_pred)
    if (missed_c, false_c, total_c) != (27, 0, 3240):
        raise SystemExit(f"Current-policy loss counts changed: {(missed_c, false_c, total_c)}")
    if (missed_t, false_t, total_t) != (7, 7, 1260):
        raise SystemExit(f"Tree loss counts changed: {(missed_t, false_t, total_t)}")

    acc_c, prec_c, rec_c = binary_rates(y_true, current_pred)
    acc_t, prec_t, rec_t = binary_rates(y_true, tree_pred)
    expected = {
        "current accuracy": (acc_c, 77.5),
        "tree accuracy": (acc_t, 88.3),
        "current precision": (prec_c, 0.0),
        "tree precision": (prec_t, 74.1),
        "current recall": (rec_c, 0.0),
        "tree recall": (rec_t, 74.1),
    }
    for name, (got, target) in expected.items():
        if abs(got - target) >= 0.05:
            raise SystemExit(f"{name} is {got:.4f}, expected {target:.1f}")
    return {
        "missed_current": missed_c,
        "false_alarm_current": false_c,
        "total_current": total_c,
        "missed_tree": missed_t,
        "false_alarm_tree": false_t,
        "total_tree": total_t,
        "avg_current": total_c / len(y_true),
        "avg_tree": total_t / len(y_true),
        "acc_current": acc_c,
        "prec_current": prec_c,
        "rec_current": rec_c,
        "acc_tree": acc_t,
        "prec_tree": prec_t,
        "rec_tree": rec_t,
    }


def assert_depth_does_not_change_calls(train, holdout, depth3_model):
    unrestricted = DecisionTreeClassifier(max_depth=None, random_state=42)
    unrestricted.fit(train[FEATURES], train[TARGET])
    same_train = np.array_equal(
        unrestricted.predict(train[FEATURES]), depth3_model.predict(train[FEATURES])
    )
    same_holdout = np.array_equal(
        unrestricted.predict(holdout[FEATURES]),
        depth3_model.predict(holdout[FEATURES]),
    )
    if not (same_train and same_holdout):
        raise SystemExit("An unrestricted tree changed at least one predicted class")
    return unrestricted


def _style_axes():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
        }
    )


def save_tree_figure(model, path):
    _style_axes()
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=list(model.classes_),
        filled=False,
        rounded=True,
        impurity=True,
        proportion=False,
        precision=3,
        fontsize=10,
        ax=ax,
    )
    ax.set_title("Clinic decision tree (v1): 280 training appointments", pad=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_cost_figure(avg_current, avg_tree, path):
    _style_axes()
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [CURRENT_POLICY, TREE_NAME]
    values = [avg_current, avg_tree]
    bars = ax.bar(labels, values, width=0.62, color=BLUE)
    ax.set_ylabel("Average loss per appointment (US$)")
    ax.set_title("Test-set estimated cost: lower is better")
    ax.set_ylim(0, 33)
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6,
            f"${value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_performance_figure(metrics, path):
    _style_axes()
    fig, ax = plt.subplots(figsize=(8.4, 5))
    labels = ["Accuracy", "No-show precision", "No-show recall"]
    current = [metrics["acc_current"], metrics["prec_current"], metrics["rec_current"]]
    tree = [metrics["acc_tree"], metrics["prec_tree"], metrics["rec_tree"]]
    x = np.arange(len(labels))
    width = 0.36
    current_bars = ax.bar(
        x - width / 2, current, width, label=CURRENT_POLICY, color=BLUE
    )
    tree_bars = ax.bar(x + width / 2, tree, width, label=TREE_NAME, color=ORANGE)
    ax.set_ylabel("Percent of test appointments (%)")
    ax.set_title("Holdout test performance (120 appointments)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper right", frameon=True)
    for bar in list(current_bars) + list(tree_bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.2,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def count_grid(frame):
    lines = [
        "| Prior no-shows | No SMS, checked-in | No SMS, no-show | SMS, checked-in | SMS, no-show |",
        "|---:|---:|---:|---:|---:|",
    ]
    for prior in range(4):
        cells = []
        for sms in (0, 1):
            for outcome in (CHECKED_IN, NO_SHOW):
                mask = (
                    (frame["prior_no_shows"] == prior)
                    & (frame["sms_reminder"] == sms)
                    & (frame["outcome"] == outcome)
                )
                cells.append(str(int(mask.sum())))
        lines.append(f"| {prior} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(train, holdout, model, metrics, tree_text, path):
    n_train = len(train)
    n_holdout = len(holdout)
    train_no_show = int((train[TARGET] == NO_SHOW).sum())
    train_checked = n_train - train_no_show
    holdout_no_show = int((holdout[TARGET] == NO_SHOW).sum())
    holdout_checked = n_holdout - holdout_no_show
    train_with_three = train.loc[train["prior_no_shows"] >= 3]
    three_without_sms = int((train_with_three["sms_reminder"] == 0).sum())
    if three_without_sms:
        three_sentence = (
            f"{three_without_sms} training patients with 3 or more prior no-shows "
            "have no SMS, so the top-right leaf can split on the reminder."
        )
    else:
        three_sentence = (
            f"All {len(train_with_three)} training patients with 3 prior no-shows "
            "have `sms_reminder` = 1, so the top-right leaf has no reminder split to learn."
        )

    text = f"""# Clinic no-show decision tree (v1)

On the 120-appointment holdout, decision tree v1 lowers average loss from **$27.00** per appointment under the current policy to **$10.50**. Accuracy moves from **77.5%** to **88.3%**. The tree recovers **74.1%** of no-shows, and **74.1%** of the visits it flags are no-shows. The current policy flags none.

The flag is a scheduling aid: a reminder call, or a cautious overbook. It is not a reason to cancel the visit.

## Data

`clinic_appointments.csv` has 400 appointments.

| Split | appointment_id | Rows | Checked-in | No-show |
|---|---:|---:|---:|---:|
| Training | 1–280 | {n_train} | {train_checked} | {train_no_show} |
| Holdout | 281–400 | {n_holdout} | {holdout_checked} | {holdout_no_show} |

Training no-show rate: {train_no_show / n_train:.1%}. Holdout no-show rate: {holdout_no_show / n_holdout:.1%}.

v1 is fit on `prior_no_shows` and `sms_reminder` only.

| Column | Role in v1 |
|---|---|
| appointment_id | Row id. 1–280 train, 281–400 holdout. |
| prior_no_shows | Model input. Earlier missed visits. Values in this file are 0, 1, 2, or 3. |
| sms_reminder | Model input. 1 means an SMS reminder was sent, 0 means it was not. |
| outcome | Target. `checked-in` or `no-show`. |
| age | Not used. Age in years. |
| lead_time_days | Not used. Days from booking to the visit. |
| weekday | Not used. Clinic day, Monday–Friday. |

This is a 400-row teaching file built so the script redraws the v1 tree and the holdout charts. It does not contain real patient records.

### Training counts

{count_grid(train)}

{three_sentence}

### Holdout counts

{count_grid(holdout)}

## The tree

Gini impurity is the split score. It is 0 when a node has one outcome and higher when the two outcomes are mixed. In each box, `value = [checked-in, no-show]` and `class` is the larger of those two counts.

![Clinic decision tree v1](outputs/decision_tree_v1.png)

Read the boxes as this rule.

Predict **no-show** when either line is true:

- `prior_no_shows` is 3 or more (in this file, 3). On the root's false branch, 15 of 17 training patients missed
- no SMS was sent, and `prior_no_shows` is 1 or 2 (29 of 52 training patients missed)

Predict **checked-in** for everyone else:

- `prior_no_shows` is 0, whether or not an SMS went out
- an SMS was sent and `prior_no_shows` is 1 or 2

The SMS branch splits once more, at 2 prior no-shows, because that group is more mixed (gini 0.43 versus 0.069). Both of those leaves still have a checked-in majority, so the extra split does not create a no-show flag. Under v1, a patient who received an SMS is flagged only when they already have 3 prior no-shows.

| Leaf | Training patients | Checked-in | No-show | Gini | Prediction |
|---|---:|---:|---:|---:|---|
| 3 prior no-shows | 17 | 2 | 15 | 0.208 | no-show |
| No SMS, 0 prior no-shows | 56 | 46 | 10 | 0.293 | checked-in |
| No SMS, 1 or 2 prior no-shows | 52 | 23 | 29 | 0.493 | no-show |
| SMS, 0 or 1 prior no-shows | 139 | 134 | 5 | 0.069 | checked-in |
| SMS, 2 prior no-shows | 16 | 11 | 5 | 0.43 | checked-in |

The same rule in text form:

```
{tree_text.rstrip()}
```

`DecisionTreeClassifier(max_depth=3)` is what keeps the drawing at five leaves. Fitting the same two columns with no depth cap splits the "no SMS, 1 or 2 prior" leaf into prior = 1 and prior = 2, and splits the "SMS, 0 or 1 prior" leaf into prior = 0 and prior = 1. In this training set each of those child pairs shares a majority class. Predicted labels on the training rows and on the holdout match the unrestricted tree, so the depth cap changes the drawing and leaves the flags alone.

## Holdout performance

The current policy predicts `checked-in` for every appointment. It is right on the {holdout_checked} patients who arrive, which is {metrics["acc_current"]:.1f}% of the holdout, and it misses all {holdout_no_show} no-shows. With zero no-show flags, no-show precision and recall are both 0.

The tree flags 27 holdout appointments. There are also 27 actual no-shows. Those two sets share 20 appointments, so precision and recall are the same ratio, 20/27 = 74.1%. The other 14 calls are split evenly: 7 false alarms and 7 missed no-shows. Correct calls: 20 + 86 = 106, which is 88.3%.

|  | Predicted checked-in | Predicted no-show |
|---|---:|---:|
| Actual checked-in | 86 | 7 |
| Actual no-show | 7 | 20 |

| Metric | Current policy | Decision tree v1 |
|---|---:|---:|
| Accuracy | {metrics["acc_current"]:.1f}% | {metrics["acc_tree"]:.1f}% |
| No-show precision | {metrics["prec_current"]:.1f}% | {metrics["prec_tree"]:.1f}% |
| No-show recall | {metrics["rec_current"]:.1f}% | {metrics["rec_tree"]:.1f}% |

![Holdout test performance](outputs/holdout_performance.png)

## Estimated cost

Loss is charged only on a wrong call.

| Call | Loss |
|---|---:|
| Predict checked-in, patient no-shows (empty slot) | ${MISSED_NO_SHOW_USD} |
| Predict no-show, patient checks in (false alarm) | ${FALSE_ALARM_USD} |
| Correct call | $0 |

Current policy: 27 missed no-shows and 0 false alarms.

- Total: 27 × ${MISSED_NO_SHOW_USD} = ${metrics["total_current"]:,.2f}
- Average: ${metrics["total_current"]:,.2f} / {n_holdout} = **${metrics["avg_current"]:.2f}**

Decision tree v1: 7 missed no-shows and 7 false alarms.

- Missed no-shows: 7 × ${MISSED_NO_SHOW_USD} = ${metrics["missed_tree"] * MISSED_NO_SHOW_USD:,.2f}
- False alarms: 7 × ${FALSE_ALARM_USD} = ${metrics["false_alarm_tree"] * FALSE_ALARM_USD:,.2f}
- Total: ${metrics["total_tree"]:,.2f}
- Average: ${metrics["total_tree"]:,.2f} / {n_holdout} = **${metrics["avg_tree"]:.2f}**

The holdout difference is ${metrics["avg_current"] - metrics["avg_tree"]:.2f} per appointment, or ${metrics["total_current"] - metrics["total_tree"]:,.2f} across the 120 visits.

![Test-set estimated cost](outputs/test_set_cost.png)

## Limits

- The holdout is the 120 rows with `appointment_id` above 280. The script does not draw a new random split; a new split would not reproduce these node counts.
- The published averages use the $120 / $60 loss above. A different price for an empty slot or a false alarm moves both bars.
- Age, lead time, and weekday stay out of v1. Adding them fits a different tree.
- Training never shows a patient with 3 prior no-shows and no SMS, so that leaf does not measure the reminder. The holdout does include both reminder values at 3 prior no-shows, and the fitted rule flags both.
- These rows are a teaching file, not a live clinic extract, and the tree is not a clinical decision system.
"""
    path.write_text(text, encoding="utf-8")


def print_summary(metrics, paths):
    print(f"Training appointments: {TRAIN_ID_MAX}")
    print(f"Holdout appointments: {HOLDOUT_SIZE}")
    print("Tree depth: 3   leaves: 5")
    print()
    print(
        f"Holdout accuracy:        {CURRENT_POLICY} {metrics['acc_current']:.1f}%"
        f" | {TREE_NAME} {metrics['acc_tree']:.1f}%"
    )
    print(
        f"No-show precision:       {CURRENT_POLICY} {metrics['prec_current']:.1f}%"
        f" | {TREE_NAME} {metrics['prec_tree']:.1f}%"
    )
    print(
        f"No-show recall:          {CURRENT_POLICY} {metrics['rec_current']:.1f}%"
        f" | {TREE_NAME} {metrics['rec_tree']:.1f}%"
    )
    print(
        f"Average loss per visit:  {CURRENT_POLICY} ${metrics['avg_current']:.2f}"
        f" | {TREE_NAME} ${metrics['avg_tree']:.2f}"
    )
    print(
        f"Holdout total loss:      {CURRENT_POLICY} ${metrics['total_current']:,.2f}"
        f" | {TREE_NAME} ${metrics['total_tree']:,.2f}"
    )
    print()
    for path in paths:
        print(f"Wrote {path.relative_to(ROOT)}")


def main():
    train, holdout = load_appointments()
    model = fit_tree(train)
    assert_published_tree(model)
    assert_depth_does_not_change_calls(train, holdout, model)

    y_holdout = holdout[TARGET].to_numpy()
    current_pred = np.full(len(holdout), CHECKED_IN)
    tree_pred = model.predict(holdout[FEATURES])
    metrics = assert_published_holdout(y_holdout, current_pred, tree_pred)

    OUTPUT_DIR.mkdir(exist_ok=True)
    tree_path = OUTPUT_DIR / "decision_tree_v1.png"
    cost_path = OUTPUT_DIR / "test_set_cost.png"
    performance_path = OUTPUT_DIR / "holdout_performance.png"
    save_tree_figure(model, tree_path)
    save_cost_figure(metrics["avg_current"], metrics["avg_tree"], cost_path)
    save_performance_figure(metrics, performance_path)

    tree_text = export_text(
        model,
        feature_names=FEATURES,
        class_names=list(model.classes_),
        decimals=1,
    )
    write_report(train, holdout, model, metrics, tree_text, REPORT_PATH)
    print_summary(metrics, [tree_path, cost_path, performance_path, REPORT_PATH])


if __name__ == "__main__":
    main()
