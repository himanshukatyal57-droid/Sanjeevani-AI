"""Traces the exact feature splits a Decision Tree follows for one input."""
from typing import List, Optional, Sequence

LEAF = -1  # scikit-learn marks a leaf with no child node


def explain_tree_prediction(model, feature_names: Sequence[str], row: List[int]) -> Optional[dict]:
    """Walk the tree from root to leaf. Returns None for a model that isn't a tree."""
    tree = getattr(model, "tree_", None)
    if tree is None:
        return None

    steps = []
    node = 0
    while tree.children_left[node] != LEAF:
        feature = tree.feature[node]
        present = row[feature] > tree.threshold[node]
        steps.append({
            "symptom": feature_names[feature],
            "present": bool(present),
            "samples": int(tree.n_node_samples[node]),
        })
        node = tree.children_right[node] if present else tree.children_left[node]

    return {
        "steps": steps,
        "leaf_node": int(node),
        "leaf_samples": int(tree.n_node_samples[node]),
        "tree_nodes": int(tree.node_count),
    }
