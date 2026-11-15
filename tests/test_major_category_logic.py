"""Unit tests for Person 2's logic. A fake embedder is used, so no model download."""

import pytest

from conftest import MAJOR_CATEGORIES

# Skip (instead of failing at collection) while Person 2's component is not in the repo.
np = pytest.importorskip("numpy")
_implementation = pytest.importorskip("app_review_intelligence.components.major_category.implementation")
_interface = pytest.importorskip("app_review_intelligence.components.major_category.interface")
classify_reviews = _implementation.classify_reviews
load_category_config = _implementation.load_category_config
classify = _interface.classify

CONFIG = {
    "other_category": "Other",
    "min_similarity": 0.5,
    "categories": [
        {"name": "Performance and Stability", "texts": ["crash slow"]},
        {"name": "User Interface and Experience", "texts": ["easy simple"]},
    ],
}
AXES = {"crash": 0, "slow": 0, "easy": 1, "simple": 1}


def fake_embed(texts):
    """Bag-of-words on two axes, L2-normalised; unknown words add nothing."""
    vectors = np.zeros((len(texts), 3))
    for i, text in enumerate(texts):
        for word in text.lower().replace(".", "").split():
            vectors[i, AXES.get(word, 2)] += 1.0 if word in AXES else 0.0
        if not vectors[i].any():
            vectors[i, 2] = 1.0
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def run(records):
    return classify_reviews(records, embed_fn=fake_embed, config=CONFIG)


def rec(rid, text):
    return {"review_id": rid, "review_text": text, "review_date": "2026-07-21"}


def test_labels_confidence_and_order():
    out = run([rec("a", "App will crash"), rec("b", "So easy"), rec("c", "hello world")])
    assert [o["review_id"] for o in out] == ["a", "b", "c"]
    assert [o["predicted_major_category"] for o in out] == [
        "Performance and Stability", "User Interface and Experience", "Other"]
    assert out[0]["confidence"] == 1.0
    assert out[2]["confidence"] == 1.0  # nothing matched at all -> sure it is Other
    assert all(0 <= o["confidence"] <= 1 for o in out)


def test_empty_text_falls_back_to_other_with_zero_confidence():
    out = run([rec("a", "   "), rec("b", None)])
    assert all(o["predicted_major_category"] == "Other" and o["confidence"] == 0.0 for o in out)


def test_empty_input_returns_empty_list():
    assert run([]) == []


def test_gold_labels_are_rejected():
    bad = rec("a", "crash")
    bad["gold_major_category"] = "User Interface and Experience"
    with pytest.raises(ValueError):
        run([bad])


def test_duplicate_ids_and_bad_types_raise():
    with pytest.raises(ValueError):
        run([rec("a", "x"), rec("a", "y")])
    with pytest.raises(TypeError):
        run("not a list")
    with pytest.raises(ValueError):
        run([{"review_id": "a"}])


def test_shipped_config_is_valid_and_has_required_categories():
    config = load_category_config()
    names = [c["name"] for c in config["categories"]]
    assert len(names) == len(set(names))
    assert "Performance and Stability" in names
    assert config["other_category"] == "Other"
    # The ten dataset "Major Class" labels = the named categories + the "Other" fallback
    # (whether or not "Other" is also listed among the categories).
    assert set(names) | {config["other_category"]} == set(MAJOR_CATEGORIES)


def test_config_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_category_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("categories: []\n")
    with pytest.raises(ValueError):
        load_category_config(bad)


def test_interface_exposes_classify():
    assert callable(classify)


# ---- optional accuracy options (model / scoring / sentence_level / smoothing) ----

def run_with(records, **options):
    return classify_reviews(records, embed_fn=fake_embed, config={**CONFIG, **options})


def test_smoothing_pulls_an_ambiguous_review_toward_its_neighbours():
    records = [rec(str(i), "crash") for i in range(10)] + [rec("odd", "easy easy crash")]
    plain = run_with(records)
    smoothed = run_with(records, smoothing={"neighbors": 3, "weight": 0.9})
    assert plain[-1]["predicted_major_category"] == "User Interface and Experience"
    assert smoothed[-1]["predicted_major_category"] == "Performance and Stability"
    assert all(o["predicted_major_category"] == "Performance and Stability" for o in smoothed[:-1])


def test_smoothing_is_skipped_for_small_batches():
    records = [rec("a", "easy easy crash"), rec("b", "crash")]
    plain = run_with(records)
    smoothed = run_with(records, smoothing={"neighbors": 5, "weight": 0.9})
    assert plain == smoothed


@pytest.mark.parametrize("scoring", ["max", "top2", "centroid"])
@pytest.mark.parametrize("sentence_level", [False, True])
def test_scoring_options_return_one_valid_prediction_per_review(scoring, sentence_level):
    records = [rec("a", "App will crash. Please fix this."), rec("b", "So easy"), rec("c", "hello world")]
    out = run_with(records, scoring=scoring, sentence_level=sentence_level)
    assert [o["review_id"] for o in out] == ["a", "b", "c"]
    assert all(0 <= o["confidence"] <= 1 for o in out)
    assert out[0]["predicted_major_category"] == "Performance and Stability"
    assert out[2]["predicted_major_category"] == "Other"


def test_new_config_options_are_validated(tmp_path):
    base = "categories:\n  - name: A\n    description: some text\n"
    for extra in ("scoring: median\n", "sentence_level: maybe\n", "model: ''\n",
                  "smoothing: 5\n", "smoothing: {neighbors: 0}\n", "smoothing: {weight: 1.5}\n"):
        path = tmp_path / "bad.yaml"
        path.write_text(base + extra)
        with pytest.raises(ValueError):
            load_category_config(path)
    good = tmp_path / "good.yaml"
    good.write_text(base + "scoring: centroid\nsentence_level: true\nsmoothing: {neighbors: 10, weight: 0.8}\n")
    config = load_category_config(good)
    assert config["scoring"] == "centroid" and config["sentence_level"] is True
    assert config["smoothing"] == {"neighbors": 10, "weight": 0.8}


# ---- optional "Other" as a competing category (other_examples / other_bias) ----

def test_other_examples_let_other_win_as_a_category_and_bias_moves_it():
    records = [rec("a", "crash"), rec("b", "easy")]
    plain = run_with(records)
    assert [o["predicted_major_category"] for o in plain] == [
        "Performance and Stability", "User Interface and Experience"]
    # "Other" has the same example as Performance: ties go to the named category...
    tied = run_with(records, other_examples=["crash"])
    assert tied[0]["predicted_major_category"] == "Performance and Stability"
    # ...unless the bias pushes "Other" ahead.
    pushed = run_with(records, other_examples=["crash"], other_bias=0.1)
    assert pushed[0]["predicted_major_category"] == "Other"
    assert pushed[1]["predicted_major_category"] == "User Interface and Experience"
    assert all(0 <= o["confidence"] <= 1 for o in pushed)


def test_other_examples_matching_text_is_other_with_valid_confidence():
    out = run_with([rec("a", "hello world")], other_examples=["hello world"])
    assert out[0]["predicted_major_category"] == "Other"
    assert out[0]["confidence"] == 1.0


def test_other_options_are_validated_and_optional(tmp_path):
    base = "categories:\n  - name: A\n    description: some text\n"
    for extra in ("other_examples: []\n", "other_examples: nope\n", "other_bias: high\n"):
        path = tmp_path / "bad.yaml"
        path.write_text(base + extra)
        with pytest.raises(ValueError):
            load_category_config(path)
    good = tmp_path / "good.yaml"
    good.write_text(base + "other_examples: [spam]\nother_bias: -0.1\n")
    config = load_category_config(good)
    assert config["other_examples"] == ["spam"] and config["other_bias"] == -0.1
    absent = tmp_path / "absent.yaml"
    absent.write_text(base)
    assert load_category_config(absent)["other_examples"] is None
