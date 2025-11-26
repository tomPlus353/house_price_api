"""Validation helpers for user-provided test inputs.

This module provides utilities to load allowed categorical values from
model artifacts (one-hot encoder and cat_cols) and to validate string
inputs (like `NearestStation`) against those allowed values.

Usage:
    from utils.validator import load_allowed_categories, validate_categorical_inputs

    allowed = load_allowed_categories("./model_extra_cat")
    cleaned, errors = validate_categorical_inputs(raw_input_dict, allowed)

The validator will return a tuple (cleaned, errors) where `errors` is a
dictionary mapping field -> list of error details. If `errors` is empty,
validation passed.
"""

from typing import Dict, List, Tuple, Any
import joblib
import difflib
import os


def load_allowed_categories(model_dir: str) -> Dict[str, List[str]]:
    """Load allowed categorical values from model artifacts.

    Expects the following files to exist in `model_dir`:
      - one_hot_encoder.joblib  (fitted sklearn OneHotEncoder)
      - cat_cols.joblib         (list of categorical column names in order)

    Returns a mapping: {column_name: [allowed_value_1, ...]}.
    If the encoder has no `categories_` attribute, an empty dict is returned.
    """
    encoder_path = os.path.join(model_dir, "one_hot_encoder.joblib")
    cat_cols_path = os.path.join(model_dir, "cat_cols.joblib")

    if not os.path.exists(encoder_path) or not os.path.exists(cat_cols_path):
        return {}

    encoder = joblib.load(encoder_path)
    cat_cols = joblib.load(cat_cols_path)

    mapping: Dict[str, List[str]] = {}

    # sklearn OneHotEncoder stores categories_ as a list matching input order
    if hasattr(encoder, "categories_") and isinstance(cat_cols, (list, tuple)):
        for i, col in enumerate(cat_cols):
            try:
                cats = encoder.categories_[i]
            except (IndexError, AttributeError):
                # categories_ may be shorter than cat_cols or missing
                cats = []
            # Ensure all values are strings for robust matching
            mapping[col] = [str(x) for x in list(cats)]

    return mapping


def suggest_closest(
    value: str, allowed: List[str], n: int = 3, cutoff: float = 0.5
) -> List[str]:
    """Return up to `n` close matches for `value` from `allowed` using difflib."""
    if not isinstance(value, str):
        return []
    return difflib.get_close_matches(value, allowed, n=n, cutoff=cutoff)


def validate_categorical_inputs(
    raw: Dict[str, Any],
    allowed_mapping: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """Validate string/categorical entries in `raw` using `allowed_mapping`.

    Parameters:
      raw: input mapping (field -> value or list-of-values). Values may be
           single values or lists; non-lists will be wrapped into lists.
      allowed_mapping: mapping from field -> list of allowed string values.

    Returns:
      cleaned: same as `raw` but all values coerced to lists of strings.
      errors: dict mapping field -> list of error dicts with keys:
          - value: the invalid value
          - suggestions: list of suggested allowed values (may be empty)

    The function only validates fields present in `allowed_mapping`. Numeric
    fields or unknown fields are left untouched in `cleaned` and not
    included in `errors`.
    """
    cleaned: Dict[str, Any] = {}
    errors: Dict[str, List[Dict[str, Any]]] = {}

    for k, v in raw.items():
        # Normalize to list
        if isinstance(v, list):
            vals = v
        else:
            vals = [v]

        # For fields that we have allowed values for, validate each entry
        if k in allowed_mapping:
            allowed = allowed_mapping.get(k, [])
            cleaned_vals: List[str] = []
            field_errors: List[Dict[str, Any]] = []

            for item in vals:
                # Convert non-None to string for comparison
                if item is None:
                    field_errors.append({"value": item, "suggestions": []})
                    continue

                s = str(item)
                if s in allowed:
                    cleaned_vals.append(s)
                else:
                    suggestions = suggest_closest(s, allowed)
                    field_errors.append({"value": s, "suggestions": suggestions})

            cleaned[k] = cleaned_vals
            if field_errors:
                errors[k] = field_errors
        else:
            # Unknown or numeric field: just coerce to list and keep original types
            cleaned[k] = vals

    return cleaned, errors


__all__ = [
    "load_allowed_categories",
    "validate_categorical_inputs",
    "suggest_closest",
]
