"""Consolidation of subcategories within a major category.

Merges duplicate or near-duplicate subcategories discovered across batches
of the same major category. Never merges across different major categories.
"""

import gc
import re
import string

import numpy as np

UNASSIGNED = "UNASSIGNED"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MERGE_SIMILARITY = 0.82


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


def _exact_name_merge(groups):
    """
    Performs Step 1: exact-name merge on normalized names.

    Args:
        groups: List of group dicts, each with 'subcategory_name',
            'description', and 'review_ids'.

    Returns:
        List of merged group dicts.
    """
    merged_by_norm = {}
    for group in groups:
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
            existing_rids = set(merged_by_norm[norm]["review_ids"])
            for rid in rids:
                if rid not in existing_rids:
                    merged_by_norm[norm]["review_ids"].append(rid)
                    existing_rids.add(rid)

    return list(merged_by_norm.values())


def _semantic_merge(
    groups,
    threshold=DEFAULT_MERGE_SIMILARITY,
    model_id=DEFAULT_EMBEDDING_MODEL,
    random_seed=42,
):
    """
    Performs Step 2: semantic merge on groups using sentence embeddings.

    Only invoked when more than one group remains after exact-name merge.
    Embeds '<name>: <description>' with MiniLM on CPU and merges with cosine
    similarity >= threshold. Avoids chaining by sorting groups by review count
    descending (ties by name) and having each candidate group join the first
    representative group whose similarity is >= threshold.

    Loads the embedding model lazily and releases it immediately after use.

    Args:
        groups: List of group dicts from Step 1.
        threshold: Cosine similarity threshold for merging.
        model_id: HuggingFace model identifier.
        random_seed: Seed for reproducibility.

    Returns:
        List of semantically merged group dicts.
    """
    if len(groups) <= 1:
        return groups

    # Sort groups by review count descending, ties by name
    sorted_groups = sorted(
        groups,
        key=lambda g: (-len(g["review_ids"]), g["subcategory_name"]),
    )

    texts = [
        f"{g['subcategory_name']}: {g['description']}"
        for g in sorted_groups
    ]

    # Lazy load sentence-transformers on CPU
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device="cpu")
    try:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    finally:
        del model
        gc.collect()

    clusters = []  # list of dicts: {'group': dict, 'embedding': np.ndarray, 'review_ids': list}

    for idx, (group, emb) in enumerate(zip(sorted_groups, embeddings)):
        matched_cluster = None
        for cluster in clusters:
            sim = float(np.dot(emb, cluster["embedding"]))
            if sim >= threshold:
                matched_cluster = cluster
                break

        if matched_cluster is not None:
            # Join existing cluster: representative's name and description win
            existing_rids = set(matched_cluster["review_ids"])
            for rid in group["review_ids"]:
                if rid not in existing_rids:
                    matched_cluster["review_ids"].append(rid)
                    existing_rids.add(rid)
        else:
            # Form a new representative cluster
            clusters.append({
                "group": group,
                "embedding": emb,
                "subcategory_name": group["subcategory_name"],
                "description": group["description"],
                "review_ids": list(group["review_ids"]),
                "norm_name": normalize_name(group["subcategory_name"]),
            })

    merged_groups = []
    for c in clusters:
        merged_groups.append({
            "subcategory_name": c["subcategory_name"],
            "description": c["description"],
            "review_ids": c["review_ids"],
            "norm_name": c["norm_name"],
        })

    return merged_groups


def consolidate_subcategories(
    major_category,
    category_code,
    raw_subcategories,
    merge_similarity=DEFAULT_MERGE_SIMILARITY,
    embedding_model_id=DEFAULT_EMBEDDING_MODEL,
    random_seed=42,
):
    """
    Consolidates subcategories within a single major category.

    Executes Step 1 (exact-name merge) followed by Step 2 (semantic merge
    if >1 group remains). Assigns final IDs formatted as <CODE>_<3 digits>
    ordered by review count descending, with ties broken by normalized name.

    Args:
        major_category: Name of the major category.
        category_code: 3-4 letter prefix code for subcategory ID.
        raw_subcategories: List of dicts with 'subcategory_name', 'description',
            and 'review_ids'.
        merge_similarity: Cosine similarity threshold for semantic merging.
        embedding_model_id: HuggingFace model ID for sentence embeddings.
        random_seed: Integer random seed for determinism.

    Returns:
        Tuple of (subcategories_list, id_mapping_dict, merge_count_int).
    """
    if not raw_subcategories:
        return [], {}, 0

    # Step 1: Exact-name merge
    groups_after_step1 = _exact_name_merge(raw_subcategories)

    # Step 2: Semantic merge (only if > 1 group remains)
    if len(groups_after_step1) > 1:
        merged_groups = _semantic_merge(
            groups_after_step1,
            threshold=merge_similarity,
            model_id=embedding_model_id,
            random_seed=random_seed,
        )
    else:
        merged_groups = groups_after_step1

    merges_count = len(raw_subcategories) - len(merged_groups)

    # Final ID assignment: order by review count descending, ties by normalized name
    merged_groups.sort(
        key=lambda g: (-len(g["review_ids"]), g["norm_name"]),
    )

    final_subcategories = []
    id_mapping = {}

    for index, group in enumerate(merged_groups, start=1):
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
