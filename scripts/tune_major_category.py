"""Find the best major-category settings (run once, locally; changes no project files).

For each embedding model it embeds the cleaned dataset once, then tries every combination of:
    * scoring:        max / top2 / centroid
    * unit:           whole review / best sentence
    * smoothing:      off, or blend with the most similar reviews (neighbors, weight)
    * threshold for "Other"
It uses the real functions from the major_category component, so whatever wins here is exactly
what classify() will do once the same settings are put in config/major_categories.yaml.

Run from the repo root (first run downloads each model, roughly 100-450 MB):
    python scripts/tune_major_category.py
    python scripts/tune_major_category.py --models sentence-transformers/all-MiniLM-L6-v2
    python scripts/tune_major_category.py --models BAAI/bge-base-en-v1.5 thenlper/gte-base

Gold labels are used only to score the results here, never to make predictions.
Settings are ranked on the even-numbered reviews and re-checked on the odd-numbered ones;
trust the HELD-OUT column, because it shows whether a gain is real or overfit.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app_review_intelligence.adapters.input.interface import load_and_prepare  # noqa: E402
from app_review_intelligence.components.major_category import implementation as impl  # noqa: E402

DATASET = ROOT / "dataset" / "processed" / "combined_reviews_output.csv"
DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
]
THRESHOLDS = [round(t, 3) for t in np.arange(0.20, 0.9001, 0.025)]
METHODS = ("max", "top2", "centroid")
UNITS = ("review", "sentence")
SMOOTHING = (None, (10, 0.5), (10, 0.8), (20, 0.8))
BIASES = [round(b, 3) for b in np.arange(-0.20, 0.0501, 0.02)]  # only used when "other_examples" is set


def metrics(pred, gold, labels):
    f1s = []
    for label in labels:
        tp = int(((pred == label) & (gold == label)).sum())
        p_count, g_count = int((pred == label).sum()), int((gold == label).sum())
        precision = tp / p_count if p_count else 0.0
        recall = tp / g_count if g_count else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    other_mask = gold == labels[-1]
    return float((pred == gold).mean()), float(np.mean(f1s)), float((pred[other_mask] == labels[-1]).mean())


def sweep_model(model_name, texts, gold, names, other, config, tune, hold):
    n_named = len(names)
    def_texts, owners = [], []
    for i, category in enumerate(config["categories"]):
        def_texts.extend(category["texts"])
        owners.extend([i] * len(category["texts"]))
    other_examples = config.get("other_examples")
    if other_examples:  # "Other" competes as one extra category (last column)
        def_texts.extend(other_examples)
        owners.extend([n_named] * len(other_examples))
    n_cat = n_named + (1 if other_examples else 0)
    biases = BIASES if other_examples else [0.0]
    owners = np.array(owners)

    print(f"\nEmbedding with {model_name} ...", flush=True)
    embed = impl._get_default_embedder(model_name)
    def_vecs = np.asarray(embed(def_texts), dtype=float)
    review_vecs = np.asarray(embed(texts), dtype=float)
    sentences, starts = [], []
    for text in texts:
        starts.append(len(sentences))
        sentences.extend(impl._split_sentences(text))
    sentence_vecs = np.asarray(embed(sentences), dtype=float)
    starts = np.array(starts)
    neighbor_cache = {k: impl._neighbors(review_vecs, k) for k in {s[0] for s in SMOOTHING if s}}

    rows = []
    for unit in UNITS:
        for method in METHODS:
            if unit == "review":
                base = impl._category_scores(review_vecs, def_vecs, owners, n_cat, method)
            else:
                sent = impl._category_scores(sentence_vecs, def_vecs, owners, n_cat, method)
                base = np.maximum.reduceat(sent, starts, axis=0)
            for smooth in SMOOTHING:
                smoothed = base if smooth is None else impl._smooth(base, *neighbor_cache[smooth[0]], smooth[1])
                for bias in biases:
                    scores = smoothed.copy()
                    if other_examples:
                        scores[:, n_named] += bias
                    best, best_score = scores.argmax(axis=1), scores.max(axis=1)
                    named = np.array(names + [other])[best]
                    for t in THRESHOLDS:
                        pred = np.where((best == n_named) | (best_score < t), other, named)
                        acc_all, f1_all, other_rec = metrics(pred, gold, names + [other])
                        rows.append((float((pred[tune] == gold[tune]).mean()), float((pred[hold] == gold[hold]).mean()),
                                     acc_all, f1_all, other_rec, model_name, unit, method, smooth, t, bias))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    records, eval_labels, _ = load_and_prepare([str(DATASET)])
    gold = np.array([l["gold_major_category"] for l in eval_labels])  # evaluation only
    texts = [r["review_text"] for r in records]
    config = impl.load_category_config()
    names, other = [c["name"] for c in config["categories"]], config["other_category"]
    tune = np.arange(len(texts)) % 2 == 0
    hold = ~tune

    rows = []
    for model_name in args.models:
        rows += sweep_model(model_name, texts, gold, names, other, config, tune, hold)

    header = f"{'model':<22}{'unit':<9}{'scoring':<10}{'smoothing':<12}{'thresh':>7}{'bias':>7}{'tune':>7}{'HELD-OUT':>10}{'all':>7}{'macroF1':>9}{'Other rec':>11}"

    def show(r):
        smooth = "off" if r[8] is None else f"k{r[8][0]} w{r[8][1]}"
        return (f"{r[5].split('/')[-1][:21]:<22}{r[6]:<9}{r[7]:<10}{smooth:<12}{r[9]:>7.3f}{r[10]:>7.2f}"
                f"{r[0]:>7.3f}{r[1]:>10.3f}{r[2]:>7.3f}{r[3]:>9.3f}{r[4]:>11.3f}")

    first = args.models[0]
    baseline = next(r for r in rows if r[5] == first and r[6] == "review" and r[7] == "max"
                    and r[8] is None and abs(r[9] - 0.30) < 1e-9 and abs(r[10]) < 1e-9)
    print("\nCurrent behaviour (review / max / no smoothing / 0.30):")
    print(header)
    print(show(baseline))

    print("\nBest setting per model (ranked on the tuning half):")
    print(header)
    for model_name in args.models:
        print(show(max((r for r in rows if r[5] == model_name), key=lambda r: r[0])))

    print("\nTop 8 overall:")
    print(header)
    for r in sorted(rows, key=lambda r: r[0], reverse=True)[:8]:
        print(show(r))

    best = max(rows, key=lambda r: r[0])
    print(f"\nBest overall: held-out accuracy {best[1]:.3f} (current behaviour: {baseline[1]:.3f})")
    print("\nPut this in config/major_categories.yaml (replace the existing min_similarity line):\n")
    print(f"model: {best[5]}")
    print(f"scoring: {best[7]}")
    print(f"sentence_level: {'true' if best[6] == 'sentence' else 'false'}")
    if best[8] is not None:
        print(f"smoothing: {{neighbors: {best[8][0]}, weight: {best[8][1]}}}")
    print(f"min_similarity: {best[9]}")
    if config.get("other_examples"):
        print(f"other_bias: {best[10]}")


if __name__ == "__main__":
    main()
