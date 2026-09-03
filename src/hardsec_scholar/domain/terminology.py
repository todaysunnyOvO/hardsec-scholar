"""Load and apply deterministic hardware-security terminology expansion."""

import re
from pathlib import Path
from typing import Any

import yaml

Terminology = dict[str, tuple[str, ...]]


def load_terminology(path: Path | str = "config/terminology.yaml") -> Terminology:
    """Load a normalized terminology mapping from YAML."""
    terminology_path = Path(path)
    if not terminology_path.is_file():
        raise FileNotFoundError(f"Terminology file does not exist: {terminology_path}")

    with terminology_path.open(encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream) or {}

    if not isinstance(raw, dict):
        raise ValueError("Terminology configuration must be a YAML mapping")

    result: Terminology = {}
    for term, expansions in raw.items():
        if not isinstance(term, str) or not term.strip():
            raise ValueError("Terminology keys must be non-empty strings")
        if not isinstance(expansions, list) or not all(
            isinstance(value, str) and value.strip() for value in expansions
        ):
            raise ValueError(f"Expansions for '{term}' must be non-empty strings")
        result[term.strip().lower()] = tuple(
            dict.fromkeys(value.strip() for value in expansions)
        )
    return result


def expand_query(
    query: str,
    terminology: Terminology,
    *,
    max_expansions: int = 8,
) -> list[str]:
    """Generate deterministic query variants for recognized domain terms."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if max_expansions < 0:
        raise ValueError("max_expansions must not be negative")

    query_variants = [normalized_query]
    lowered = normalized_query.lower()
    additions: list[str] = []

    for term, expansions in terminology.items():
        term_pattern = rf"(?<![\w-]){re.escape(term)}(?![\w-])"
        term_present = re.search(term_pattern, lowered, flags=re.IGNORECASE) is not None
        expansion_present = any(
            expansion.lower() in lowered for expansion in expansions
        )
        if not term_present and not expansion_present:
            continue

        for expansion in expansions:
            if expansion.lower() not in lowered and expansion not in additions:
                additions.append(expansion)

    for expansion in additions[:max_expansions]:
        query_variants.append(f"{normalized_query} {expansion}")
    return query_variants
