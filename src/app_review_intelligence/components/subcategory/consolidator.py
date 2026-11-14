"""Consolidation of subcategories within a major category.

Merges duplicate or near-duplicate subcategories discovered across batches
of the same major category. Never merges across different major categories.
"""

import copy
import re
import string

UNASSIGNED = "UNASSIGNED"


def normalize_name(name):
    """
    Normalizes subcategory name for exact matching.

    Lowercases, strips leading/trailing whitespace, removes punctuation,
    and collapses internal whitespace.

    Args:
        name: Raw subcategory name string.

    Returns:
        Normalized string.
    """
    text = name.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def consolidate_subcategories(
    major_category,
    category_code,
    raw_subcategories,
    merge_similarity=0.82,
    embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
    random_seed=42,
):
    """
    Consolidates subcategories for a single major category.

    Step 1: Exact-name merge on normalized names.
    Assigns final IDs formatted as <CODE>_<3 digits> ordered by review count descending,
    with ties broken by normalized name.

    Args:
        major_category: Name of the major category.
        category_code: 3-4 letter code for ID prefix (e.g. 'PERF').
        raw_subcategories: List of dicts with 'subcategory_name', 'description',
            and 'review_ids'.
        merge_similarity: Cosine similarity threshold for semantic merge.
        embedding_model_id: HuggingFace model ID for sentence embeddings.
        random_seed: Random seed for determinism.

    Returns:
        Tuple of (subcategories_list, id_mapping_dict, merge_count_int).
    """
    if not raw_subcategories:
        return [], {}, 0

    # Step 1: Exact-name merge
    merged_by_norm = {}
    for group in raw_subcategories:
        name = group.get("subcategory_name", "").strip()
        desc = group.get("description", "").strip()
        rids = list(group.get("review_ids", []))
        if not rids:
            continue

        norm = normalize_name(name)
        if norm not in merged_by_norm:
            merged_by_norm[norm] = {
                "subcategory_name": name,
                "description": desc,
                "review_ids": list(rids),
                "norm_name": norm,
            }
        else:
            # Merge review IDs preserving uniqueness and order
            existing_rids = set(merged_by_norm[norm]["review_ids"])
            for rid in rids:
                if rid not in existing_rids:
                    merged_by_norm[norm]["review_ids"].append(rid)
                    existing_rids.add(rid)

    groups_after_step1 = list(merged_by_norm.values())
    merges_count = len(raw_subcategories) - len(groups_after_step1)

    # Sort merged groups: review count descending, ties by normalized name
    groups_after_step1.sort(
        key=lambda g: (-len(g["review_ids"]), g["norm_name"])
    )

    final_subcategories = []
    id_mapping = {}

    for index, group in enumerate(groups_after_step1, start=1):
        sub_id = f"{category_code}_{index:03d}"
        final_subcategories.append({
            "subcategory_id": sub_id,
            "subcategory_name": group["subcategory_name"],
            "description": group["description"],
            "major_category": major_category,
        })
        for rid in group["review_ids"]:
            id_mapping[rid] = sub_id

    return final_subcategories, id_mapping, merges_count
