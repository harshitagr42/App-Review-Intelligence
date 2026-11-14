"""Accuracy of major-category prediction on the real combined dataset.

Runs the real MiniLM model, so it is skipped when the dataset or
sentence-transformers is missing. Needs the dataset at
dataset/processed/combined_reviews_output.csv (gitignored, so it never runs in CI).

This test only reports the accuracy; it never fails on a low number.

Where the output goes:
    * printed to the terminal at the end of the test (no -s needed)
    * saved to outputs/major_category_accuracy.txt (gitignored)

Run only this test:
    pytest tests/test_major_category_accuracy.py
"""

from pathlib import Path

import pandas as pd
import pytest

from conftest import MAJOR_CATEGORIES, get_component_function

ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / "dataset" / "processed" / "combined_reviews_output.csv"
REPORT_PATH = ROOT / "outputs" / "major_category_accuracy.txt"


def build_report(labels, predictions):
    """Return (accuracy, report_text). Pure function: no model, no files."""
    gold = {l["review_id"]: l["gold_major_category"] for l in labels}
    rows = [
        {"gold": gold[p["review_id"]], "pred": p["predicted_major_category"]}
        for p in predictions
    ]
    df = pd.DataFrame(rows)
    accuracy = float((df.gold == df.pred).mean())

    lines = [f"Reviews evaluated: {len(df)}", f"Overall accuracy: {accuracy:.3f}", ""]
    lines.append(f"{'Category':<36}{'Support':>8}{'Precision':>11}{'Recall':>9}")
    for category in MAJOR_CATEGORIES:
        support = int((df.gold == category).sum())
        predicted = int((df.pred == category).sum())
        correct = int(((df.gold == category) & (df.pred == category)).sum())
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        lines.append(f"{category:<36}{support:>8}{precision:>11.3f}{recall:>9.3f}")

    table = pd.crosstab(df.gold, df.pred).reindex(
        index=MAJOR_CATEGORIES, columns=MAJOR_CATEGORIES, fill_value=0
    )
    lines += ["", "Confusion table (rows = gold, columns = predicted):", table.to_string()]
    return accuracy, "\n".join(lines)


@pytest.mark.skipif(not COMBINED.exists(), reason="local dataset not available")
def test_major_category_accuracy_on_dataset(capsys):
    pytest.importorskip("sentence_transformers")
    load_and_prepare = get_component_function(
        "app_review_intelligence.adapters.input.interface", "load_and_prepare"
    )
    classify = get_component_function(
        "app_review_intelligence.components.major_category.interface", "classify"
    )

    records, labels, _ = load_and_prepare([str(COMBINED)])
    predictions = classify(records)  # model-facing records only: no gold labels

    assert [p["review_id"] for p in predictions] == [r["review_id"] for r in records]

    accuracy, report = build_report(labels, predictions)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    with capsys.disabled():
        print("\n" + report + f"\n\nSaved to {REPORT_PATH}")
        print(f"\nMajor-category accuracy: {accuracy:.3f}")
