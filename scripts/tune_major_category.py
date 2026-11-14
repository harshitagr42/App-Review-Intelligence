"""Find the best major-category settings for all-MiniLM-L6-v2 only (run locally; changes no project files).

It embeds the cleaned dataset once with MiniLM, then tries every combination of:
    * scoring:        max / top2 / centroid
    * unit:           whole review / best sentence
    * smoothing:      off, or blend with the most similar reviews (neighbors, weight)
    * "Other" bias    (only when config/major_categories.yaml has other_examples)
    * threshold for "Other"
It uses the real functions from the major_category component, so whatever wins here is exactly
what classify() will do once the same settings are put in config/major_categories.yaml.

Run from the repo root:
    python scripts/tune_minilm.py

Gold labels are used only to score the results here, never to make predictions.
Settings are ranked on the even-numbered reviews and re-checked on the odd-numbered ones;
trust the HELD-OUT column, because it shows whether a gain is real or overfit.
The "all" column is accuracy over every review, the same number the accuracy test reports,
so it is the one to compare against the mpnet result.

The full output is also saved to outputs/minilm_tuning.txt (outputs/ is gitignored).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app_review_intelligence.adapters.input.interface import load_and_prepare  # noqa: E402
from app_review_intelligence.components.major_category import implementation as impl  # noqa: E402

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASET = ROOT / "dataset" / "processed" / "combined_reviews_output.csv"
RESULT_PATH = ROOT / "outputs" / "minilm_tuning.txt"

THRESHOLDS = [round(t, 3) for t in np.arange(0.20, 0.9001, 0.025)]
METHODS = ("max", "top2", "centroid")
UNITS = ("review", "sentence")
SMOOTHING = (None, (10, 0.5), (10, 0.8), (20, 0.8))
BIASES = [round(b, 3) for b in np.arange(-0.20, 0.0501, 0.02)]  # only used when "other_examples" is set

HEADER = (
    f"{'unit':<9}{'scoring':<10}{'smoothing':<12}{'thresh':>7}{'bias':>7}"
    f"{'tune':>7}{'HELD-OUT':>10}{'all':>7}{'macroF1':>9}{'Other rec':>11}"
)


def metrics(pred, gold, labels):
    """Return (accuracy, macro F1, recall of the last label, which is "Other")."""
    f1s = []
    for label in labels:
        tp = int(((pred == label) & (gold == label)).sum())
        p_count, g_count = int((pred == label).sum()), int((gold == label).sum())
        precision = tp / p_count if p_count else 0.0
        recall = tp / g_count if g_count else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    other_mask = gold == labels[-1]
    return float((pred == gold).mean()), float(np.mean(f1s)), float((pred[other_mask] == labels[-1]).mean())


def sweep(texts, gold, names, other, config, tune, hold):
    """Try every setting combination. Each row is a dict so the columns cannot get mixed up."""
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

    print(f"Embedding with {MODEL_NAME} ...", flush=True)
    embed = impl._get_default_embedder(MODEL_NAME)
    def_vecs = np.asarray(embed(def_texts), dtype=float)
    review_vecs = np.asarray(embed(texts), dtype=float)
    sentences, starts = [], []
    for text in texts:
        starts.append(len(sentences))
        sentences.extend(impl._split_sentences(text))
    sentence_vecs = np.asarray(embed(sentences), dtype=float)
    starts = np.array(starts)
    neighbor_cache = {k: impl._neighbors(review_vecs, k) for k in {s[0] for s in SMOOTHING if s}}

    labels = names + [other]
    label_array = np.array(labels)
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
                    named = label_array[best]
                    for t in THRESHOLDS:
                        pred = np.where((best == n_named) | (best_score < t), other, named)
                        acc_all, f1_all, other_rec = metrics(pred, gold, labels)
                        rows.append({
                            "tune": float((pred[tune] == gold[tune]).mean()),
                            "held": float((pred[hold] == gold[hold]).mean()),
                            "all": acc_all, "f1": f1_all, "other_rec": other_rec,
                            "unit": unit, "method": method, "smooth": smooth, "t": t, "bias": bias,
                        })
    return rows, bool(other_examples)


def show(r):
    smooth = "off" if r["smooth"] is None else f"k{r['smooth'][0]} w{r['smooth'][1]}"
    return (f"{r['unit']:<9}{r['method']:<10}{smooth:<12}{r['t']:>7.3f}{r['bias']:>7.2f}"
            f"{r['tune']:>7.3f}{r['held']:>10.3f}{r['all']:>7.3f}{r['f1']:>9.3f}{r['other_rec']:>11.3f}")


def main():
    if not DATASET.exists():
        sys.exit(f"Dataset not found: {DATASET}\nRun this from the machine that has dataset/processed/.")

    records, eval_labels, _ = load_and_prepare([str(DATASET)])
    gold = np.array([l["gold_major_category"] for l in eval_labels])  # evaluation only
    texts = [r["review_text"] for r in records]
    config = impl.load_category_config()
    names, other = [c["name"] for c in config["categories"]], config["other_category"]
    tune = np.arange(len(texts)) % 2 == 0
    hold = ~tune

    rows, has_other_examples = sweep(texts, gold, names, other, config, tune, hold)

    out = []
    baseline = next(r for r in rows if r["unit"] == "review" and r["method"] == "max"
                    and r["smooth"] is None and abs(r["t"] - 0.30) < 1e-9 and abs(r["bias"]) < 1e-9)
    out += [f"MiniLM tuning over {len(texts)} reviews, {len(rows)} setting combinations tried.", "",
            "Plain baseline (review / max / no smoothing / 0.30):", HEADER, show(baseline), "",
            "Best threshold/bias for each unit + scoring + smoothing choice, top 8 (ranked on the tuning half):",
            HEADER]
    ranked = sorted(rows, key=lambda r: r["tune"], reverse=True)
    seen, distinct = set(), []
    for r in ranked:
        key = (r["unit"], r["method"], r["smooth"])
        if key not in seen:
            seen.add(key)
            distinct.append(r)
    out += [show(r) for r in distinct[:8]]

    best = ranked[0]
    out += ["", f"Best: held-out accuracy {best['held']:.3f}, accuracy on all reviews {best['all']:.3f} "
                f"(plain baseline held-out: {baseline['held']:.3f})",
            "", "To try it, put this in config/major_categories.yaml (replace the matching lines):", "",
            f"model: {MODEL_NAME}",
            f"scoring: {best['method']}",
            f"sentence_level: {'true' if best['unit'] == 'sentence' else 'false'}"]
    if best["smooth"] is not None:
        out.append(f"smoothing: {{neighbors: {best['smooth'][0]}, weight: {best['smooth'][1]}}}")
    out.append(f"min_similarity: {best['t']}")
    if has_other_examples:
        out.append(f"other_bias: {best['bias']}")

    text = "\n".join(out)
    print("\n" + text)
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()