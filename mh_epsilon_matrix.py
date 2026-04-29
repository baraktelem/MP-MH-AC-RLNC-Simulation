"""
Paper-style ε matrix helpers for Cohen et al. MP-MH (Fig. 19 / Sec. V-C).

Article convention: rows = paths (Path 1…P), columns = hops (Hop 1…H).
MpMhNetwork expects path_epsilons[hop_idx][path_idx] (hop-major).
"""

from __future__ import annotations

# Default topology matching the paper's 4×3 example matrix
DEFAULT_NUM_PATHS = 4
DEFAULT_NUM_HOPS = 3


def article_matrix_paper(e1: float, e2: float) -> list[list[float]]:
    """
    Return ε matrix with paths as rows and hops as columns (0-based indices).

    Path row i corresponds to article Path i+1; column j to Hop j+1.
    """
    # MP-MH
    return [
        [e1, 0.6, 0.3],
        [0.8, e1, e1],
        [0.2, e2, 0.7],
        [e2, 0.4, e2],
    ]

    # MP 
    # return [[e1], [e2], [0.2], [0.8]]


def validate_article_matrix(
    article: list[list[float]], num_paths: int, num_hops: int
) -> None:
    if len(article) != num_paths:
        raise ValueError(
            f"Epsilon matrix must have {num_paths} rows (paths), got {len(article)}"
        )
    for i, row in enumerate(article):
        if len(row) != num_hops:
            raise ValueError(
                f"Row {i} must have length {num_hops} (hops), got {len(row)}"
            )


def hop_major_from_article(article: list[list[float]]) -> list[list[float]]:
    """Transpose path×hop (article) → hop×path for MpMhNetwork."""
    num_paths = len(article)
    num_hops = len(article[0]) if article else 0
    validate_article_matrix(article, num_paths, num_hops)
    return [
        [article[p][h] for p in range(num_paths)] for h in range(num_hops)
    ]


def build_path_epsilons_paper(e1: float, e2: float) -> list[list[float]]:
    """Hop-major path_epsilons for MpMhNetwork (paper 4×3 template)."""
    return hop_major_from_article(article_matrix_paper(e1, e2))


def article_matrix_for_hops(e1: float, e2: float, num_hops_eff: int) -> list[list[float]]:
    """
    Article ε matrix for MpMhNetwork: full paper 3-hop template, or Hop 1 only (4×1)
    taken from the first column of that template when num_hops_eff == 1.
    """
    full = article_matrix_paper(e1, e2)
    if num_hops_eff == len(full[0]):
        return full
    if num_hops_eff == 1:
        return [[full[p][0]] for p in range(len(full))]
    raise ValueError(
        f"Unsupported num_hops_eff={num_hops_eff}; use 1 or {len(full[0])} for this template."
    )


def build_path_epsilons(e1: float, e2: float, num_hops_eff: int) -> list[list[float]]:
    """Hop-major path_epsilons for MpMhNetwork (paper-derived template)."""
    return hop_major_from_article(article_matrix_for_hops(e1, e2, num_hops_eff))


def format_cell(v: float) -> str:
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.1f}"


def print_article_epsilon_matrix(
    e1: float,
    e2: float,
    num_paths: int = DEFAULT_NUM_PATHS,
    num_hops: int = DEFAULT_NUM_HOPS,
) -> None:
    """Print matrix with article labels Path 1…P, Hop 1…H."""
    article = article_matrix_for_hops(e1, e2, num_hops)
    validate_article_matrix(article, num_paths, num_hops)

    hop_headers = "".join(f"{'Hop ' + str(h + 1):>12}" for h in range(num_hops))
    print(f"\nε matrix (rows = Path 1…{num_paths}, cols = Hop 1…{num_hops})")
    print(f"{'Path':>6}{hop_headers}")
    for p in range(num_paths):
        row = article[p]
        cells = "".join(f"{format_cell(row[h]):>12}" for h in range(num_hops))
        print(f"{p + 1:>6}{cells}")
    print(f"  (ε₁={e1}, ε₂={e2})\n")
