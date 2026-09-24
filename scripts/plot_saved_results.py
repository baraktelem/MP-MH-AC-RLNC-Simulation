"""
Script to load and plot saved simulation results with comparison support.

Usage:
    # Plot single file:
    python plot_saved_results.py simulation_results.pkl
    
    # Compare multiple protocols:
    python plot_saved_results.py protocol1.pkl protocol2.pkl protocol3.pkl
    
    # Labels can be auto-generated from filenames or you can specify them
"""
import sys
import os
import pickle
import re
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "jamming_simulation"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from mp_simulation import plot_stats, aggregate_results
from jam_mp_mh_simulation import plot_per_k_surfaces, network_min_cut_capacity
from sr_arq_simulation import aggregate as sr_aggregate, plot_compare as sr_plot_compare


# ---------------------------------------------------------------------------
# Format detection + loading (flat list, jam per-k dict, multi-series dict)
# ---------------------------------------------------------------------------

# Friendly labels for series keys used by sr_arq_* simulation pickles.
_SERIES_LABELS = {
    "best": "SR-ARQ best single path",
    "matched": "SR-ARQ P matched paths",
    "sr": "SR ARQ (shared multipath)",
    "sr_indep": "SR ARQ (coupled round-robin)",
    "sr_perpath": "SR ARQ (independent per-path)",
    "sr_mpmh": "SR ARQ (MP-MH hop-by-hop)",
    "ac": "MP AC-RLNC",
}


def is_per_k_results(obj) -> bool:
    """True when obj is {series_value: [(e1, e2, stats), ...]} as saved by the
    jam surface sweeps: sweep_eps_grid_per_k (int k keys) and
    sweep_eps_grid_per_alpha (numeric alpha keys, which may be float, e.g. 1/RTT).
    Both share the same structure; callers distinguish k vs alpha by filename
    (see series_kind_for_file).

    Tolerates empty value lists (a partial pickle written by the incremental
    per-series save during a run that hasn't yet started one or more values).
    Requires at least one series value to have non-empty data so the format can
    be distinguished from arbitrary {number: list} dicts.
    """
    if not isinstance(obj, dict) or not obj:
        return False
    saw_any_data = False
    for key, value in obj.items():
        # Accept int/float series keys (k or alpha); reject bool (a subclass of int).
        if isinstance(key, bool) or not isinstance(key, (int, float)):
            return False
        if not isinstance(value, list):
            return False
        if not value:
            continue  # partial pickle: this series value hasn't started yet
        saw_any_data = True
        first = value[0]
        if not isinstance(first, tuple) or len(first) != 3:
            return False
    return saw_any_data


def series_kind_for_file(filepath) -> tuple[str, callable]:
    """Infer the series dimension of a numeric-keyed surface pickle from its
    filename: sweep_eps_grid_per_alpha -> 'α', per_p -> 'P', per_h -> 'H',
    everything else (per_k) -> 'k'. For alpha files the RTT is parsed from the
    filename when present so the legend can show the jamming-round length
    (jamming_round_time = RTT / alpha), matching the live sim plot."""
    stem = os.path.splitext(os.path.basename(str(filepath)))[0].lower()
    if ("per_alpha" in stem) or ("sweep_alpha" in stem):
        m = re.search(r"rtt[_]?(\d+)", stem)
        rtt = int(m.group(1)) if m else None
        if rtt:
            return "α", (lambda a: f"{a:g} (round≈{rtt / a:g})")
        return "α", (lambda a: f"{a:g}")
    if "per_p" in stem:
        return "P", (lambda v: f"{v}")
    if "per_h" in stem:
        return "H", (lambda v: f"{v}")
    return "k", (lambda v: f"{v}")


def is_multi_series_results(obj) -> bool:
    """True when obj is {series_name: [(e1, e2, stats), ...]} as saved by
    sr_arq_simulation / sr_arq_mpmh_simulation (string keys, not int k)."""
    if not isinstance(obj, dict) or not obj:
        return False
    if is_per_k_results(obj):
        return False
    saw_any_data = False
    for key, value in obj.items():
        if not isinstance(key, str) or not isinstance(value, list):
            return False
        if not value:
            continue
        saw_any_data = True
        first = value[0]
        if not isinstance(first, tuple) or len(first) != 3:
            return False
    return saw_any_data


def series_label(key: str, file_stem: str | None = None) -> str:
    """Map a pickle series key to a plot legend label."""
    if key in _SERIES_LABELS:
        return _SERIES_LABELS[key]
    if file_stem:
        return f"{file_stem}: {key}"
    return key


def load_results_any(filename: str):
    """Load a pickle; supports flat list, jam per-k dict, and multi-series
    dict (sr_arq_* results_by_proto / results_by_setting) formats."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file not found: {filename}")
    with open(filename, "rb") as f:
        results = pickle.load(f)

    print(f"[OK] Results loaded from: {filename}")
    if is_per_k_results(results):
        non_empty = {k: v for k, v in results.items() if v}
        empty_keys = sorted(set(results) - set(non_empty))
        total = sum(len(v) for v in non_empty.values())
        print(
            f"     Per-series surface format (per-k / per-alpha): {len(results)} "
            f"series value(s) ({len(non_empty)} non-empty, {len(empty_keys)} empty), "
            f"{total} total data points"
        )
        if empty_keys:
            print(f"     Empty series value(s) (skipped in plot): {empty_keys}")
    elif is_multi_series_results(results):
        non_empty = {k: v for k, v in results.items() if v}
        total = sum(len(v) for v in non_empty.values())
        keys = ", ".join(non_empty.keys())
        print(
            f"     Multi-series format: {len(non_empty)} series ({keys}), "
            f"{total} total data points"
        )
    elif isinstance(results, list):
        print(f"     Total data points: {len(results)}")
    else:
        print(f"     Unrecognized format: {type(results).__name__}")
    return results


def _snap_agg_keys(
    agg: dict[tuple[float, float], dict], ndigits: int = 2
) -> dict[tuple[float, float], dict]:
    """Snap (e1, e2) keys to ndigits decimals so pickles produced with
    np.arange-derived floats (which drift to e.g. 0.30000000000000004) align
    with pickles produced via rounded `round(float(v), 2)` floats. Without
    this, a set-union over their keys yields phantom grid points and
    plot_surface produces spike artifacts at the misaligned cells."""
    snapped: dict[tuple[float, float], dict] = {}
    for (e1, e2), v in agg.items():
        snapped[(round(float(e1), ndigits), round(float(e2), ndigits))] = v
    return snapped


def aggregate_per_k_results(
    per_k_results: dict[int, list],
) -> dict[int, dict[tuple[float, float], dict]]:
    """Aggregate per-k results, dropping any k whose result list is empty
    (which happens with a partial pickle from an interrupted/in-progress run).
    Snaps (e1, e2) keys to 2 decimals to keep grids stable across pickles."""
    return {k: _snap_agg_keys(aggregate_results(rs)) for k, rs in per_k_results.items() if rs}


def infer_eps_grid(
    aggregated_per_k: dict[int, dict[tuple[float, float], dict]],
) -> tuple[list[float], list[float]]:
    eps1 = sorted({e1 for agg in aggregated_per_k.values() for e1, _ in agg})
    eps2 = sorted({e2 for agg in aggregated_per_k.values() for _, e2 in agg})
    return eps1, eps2


def plot_per_k_stats(
    per_k_results: dict[int, list],
    *,
    title_suffix: str,
    plot_path: str,
    num_hops_eff: int = 3,
    series_label: str = "k",
    series_fmt=None,
) -> None:
    aggregated_per_k = aggregate_per_k_results(per_k_results)
    if not aggregated_per_k:
        print(f"[ERROR] No completed {series_label} values to plot.")
        return
    eps1, eps2 = infer_eps_grid(aggregated_per_k)
    # Use the same NetworkX min-cut as mp_mh_simulation.py so the reference
    # surface is directly comparable. None/no surface if NetworkX missing.
    cap_value = network_min_cut_capacity(eps1[0], eps2[0], num_hops_eff)
    capacity_func = (
        (lambda e1, e2: network_min_cut_capacity(e1, e2, num_hops_eff))
        if cap_value is not None else None
    )
    plot_per_k_surfaces(
        aggregated_per_k,
        eps_values_e1=eps1,
        eps_values_e2=eps2,
        series_label=series_label,
        series_fmt=series_fmt,
        title_suffix=title_suffix,
        plot_path=plot_path,
        capacity_func=capacity_func,
        capacity_label="Min-cut capacity (NetworkX, layered BEC)",
    )


# ---------------------------------------------------------------------------
# Combined overlay (mix of flat-list and per-k pickles on the same axes)
# ---------------------------------------------------------------------------

def plot_overlay_surfaces(
    series: list[tuple[str, dict[tuple[float, float], dict]]],
    *,
    eps_values_e1: list[float],
    eps_values_e2: list[float],
    title_suffix: str,
    plot_path: str,
    capacity_surfaces: list[tuple[str, callable, str]] | None = None,
) -> None:
    """3 subplots (throughput / mean delay / max delay). Each entry in `series`
    contributes one alpha-blended surface per subplot in a distinct tab10 color
    with a legend entry. `capacity_surfaces` (optional) is a list of
    (label, func(e1,e2)->float, color) tuples added only to the throughput
    subplot as transparent reference surfaces."""
    n_e1 = len(eps_values_e1)
    n_e2 = len(eps_values_e2)
    EPS1, EPS2 = np.meshgrid(eps_values_e1, eps_values_e2)

    cmap = plt.get_cmap("tab10")
    color_for = lambda i: cmap(i % 10)

    metrics = [
        ("throughput_mean", "Normalized Throughput"),
        ("delay_mean_mean", "Mean In-Order Delay"),
        ("delay_max_mean", "Max In-Order Delay"),
    ]

    fig = plt.figure(figsize=(20, 6))
    for subplot_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = fig.add_subplot(1, 3, subplot_idx + 1, projection="3d")
        max_z = 0.0
        legend_patches = []

        for s_idx, (label, agg) in enumerate(series):
            grid = np.zeros((n_e1, n_e2))
            for i, e1 in enumerate(eps_values_e1):
                for j, e2 in enumerate(eps_values_e2):
                    grid[i, j] = agg.get((e1, e2), {}).get(metric_key, 0.0)
            max_z = max(max_z, float(np.max(grid)))
            color = color_for(s_idx)
            ax.plot_surface(
                EPS1, EPS2, grid.T,
                color=color, edgecolor="none", alpha=0.4,
            )
            from matplotlib.patches import Patch
            legend_patches.append(Patch(color=color, label=label, alpha=0.4))

        if subplot_idx == 0 and capacity_surfaces:
            for cap_label, cap_func, cap_color in capacity_surfaces:
                cap_grid = np.zeros((n_e1, n_e2))
                for i, e1 in enumerate(eps_values_e1):
                    for j, e2 in enumerate(eps_values_e2):
                        cap_grid[i, j] = float(cap_func(e1, e2))
                max_z = max(max_z, float(np.max(cap_grid)))
                ax.plot_surface(
                    EPS1, EPS2, cap_grid.T,
                    color=cap_color, edgecolor="none", alpha=0.2,
                )
                from matplotlib.patches import Patch
                legend_patches.append(Patch(color=cap_color, label=cap_label, alpha=0.2))

        ax.set_xlabel("ε₁")
        ax.set_ylabel("ε₂")
        ax.set_zlabel(metric_label)
        ax.set_title(metric_label)
        ax.set_zlim(0, max(0.1, max_z * 1.1))
        ax.view_init(elev=20, azim=45)
        ax.legend(handles=legend_patches, fontsize=8, loc="upper left")

    fig.suptitle(title_suffix, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Plot saved to: {plot_path}")
    plt.show()


def _eps_grid_from_aggs(aggs: list[dict[tuple[float, float], dict]]) -> tuple[list[float], list[float]]:
    eps1 = sorted({e1 for agg in aggs for e1, _ in agg})
    eps2 = sorted({e2 for agg in aggs for _, e2 in agg})
    return eps1, eps2


def build_overlay_series_and_capacities(
    flat_datasets: list[tuple[str, list]],
    per_k_datasets: list[tuple[str, dict, str]],
    *,
    num_hops_eff: int = 3,
) -> tuple[list[tuple[str, dict]], list[tuple[str, callable, str]]]:
    """Build (series, capacity_surfaces) from raw loaded datasets."""
    series: list[tuple[str, dict]] = []
    for label, results in flat_datasets:
        series.append((label, _snap_agg_keys(aggregate_results(results))))
    for label, per_k_results, filepath in per_k_datasets:
        s_label, s_fmt = series_kind_for_file(filepath)
        for k in sorted(per_k_results.keys()):
            if not per_k_results[k]:
                continue
            agg = _snap_agg_keys(aggregate_results(per_k_results[k]))
            series.append((f"{label} {s_label}={s_fmt(k)}", agg))

    # Single shared capacity surface -- the same NetworkX min-cut used by
    # mp_mh_simulation.py. Both protocols share this reference because the
    # underlying graph capacity is identical; only the protocol's ability to
    # exploit it differs.
    capacity_surfaces: list[tuple[str, callable, str]] = []
    if (flat_datasets or per_k_datasets) and network_min_cut_capacity(0.1, 0.1, num_hops_eff) is not None:
        capacity_surfaces.append((
            "Min-cut capacity (NetworkX, layered BEC)",
            lambda e1, e2: network_min_cut_capacity(e1, e2, num_hops_eff),
            "red",
        ))

    return series, capacity_surfaces


# ---------------------------------------------------------------------------
# Paper-format comparison (stacked rows: one pkl per row, 3 metric columns)
# ---------------------------------------------------------------------------

# (mean_key, std_key, side_label). The side_label becomes the z-axis (side)
# label of each subplot -- there is no title above in paper mode.
_PAPER_METRICS = [
    ("throughput_mean", "throughput_std", "Normalized Throughput"),
    ("delay_mean_mean", "delay_mean_std", "Mean In-Order Delay [slots]"),
    ("delay_max_mean", "delay_max_std", "Max In-Order Delay [slots]"),
]


def _parse_cli_args(argv: list[str]) -> tuple[list[str], bool, str, list[str]]:
    """Split raw argv into (files, paper_mode, out_stem, row_labels).

    Recognizes the ``-paper``/``--paper`` flag, an optional output override
    (``-o``/``--out PATH`` or ``--out=PATH``), and optional per-row protocol
    headers (``--row-labels "MH MP AC-RLNC;SR ARQ"`` or ``--row-labels=...``;
    semicolon-separated, one per pkl in file order). Every other token is treated
    as a results file. ``out_stem`` is the output path without extension (paper
    mode writes both ``<stem>.png`` and ``<stem>.pdf``); defaults to
    'paper_comparison'. ``row_labels`` is empty when the flag is not given
    (headers are then auto-detected from filenames)."""
    paper_mode = False
    out_stem = "paper_comparison"
    files: list[str] = []
    row_labels: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        low = arg.lower()
        if low in ("-paper", "--paper"):
            paper_mode = True
        elif low in ("-o", "--out"):
            if i + 1 < len(argv):
                out_stem = os.path.splitext(argv[i + 1])[0]
                i += 1
            else:
                print("[WARNING] --out given without a path; using default output name.")
        elif low.startswith("--out="):
            out_stem = os.path.splitext(arg.split("=", 1)[1])[0]
        elif low in ("--row-labels", "-row-labels"):
            if i + 1 < len(argv):
                row_labels = [s.strip() for s in argv[i + 1].split(";")]
                i += 1
            else:
                print("[WARNING] --row-labels given without a value; ignoring.")
        elif low.startswith("--row-labels="):
            row_labels = [s.strip() for s in arg.split("=", 1)[1].split(";")]
        else:
            files.append(arg)
        i += 1
    return files, paper_mode, out_stem, row_labels


def _auto_protocol_label(stem: str) -> str:
    """Best-effort protocol name for a per-row header when --row-labels is not
    supplied. SR-ARQ result pickles are saved with an 'sr' filename prefix; all
    other jam sweeps come from the MH MP AC-RLNC simulator."""
    return "SR ARQ" if stem.lower().startswith("sr") else "MH MP AC-RLNC"


def build_paper_rows(
    valid_files: list[str],
) -> list[tuple[str, list[tuple[str, dict[tuple[float, float], dict]]]]]:
    """Load each pickle and turn it into one figure 'row'.

    Returns a list of ``(pkl_stem, surfaces)`` where ``surfaces`` is a list of
    ``(surface_label, aggregated_dict)``. Supports the three saved formats:
      - per-k / per-alpha dict -> one surface per non-empty series value
        (label like ``k=3`` / ``α=...``)
      - multi-series dict       -> one surface per string key (friendly label)
      - flat list               -> a single surface labelled with the stem
    Empty series / files are skipped with a warning."""
    rows: list[tuple[str, list[tuple[str, dict]]]] = []
    for filepath in valid_files:
        stem = os.path.splitext(os.path.basename(str(filepath)))[0]
        results = load_results_any(filepath)
        surfaces: list[tuple[str, dict]] = []
        if is_per_k_results(results):
            s_label, s_fmt = series_kind_for_file(filepath)
            for k in sorted(results.keys()):
                if not results[k]:
                    continue
                agg = _snap_agg_keys(aggregate_results(results[k]))
                surfaces.append((f"{s_label}={s_fmt(k)}", agg))
        elif is_multi_series_results(results):
            for key, series_results in results.items():
                if not series_results:
                    continue
                agg = _snap_agg_keys(aggregate_results(series_results))
                surfaces.append((series_label(key, stem), agg))
        elif isinstance(results, list):
            surfaces.append((stem, _snap_agg_keys(aggregate_results(results))))
        else:
            print(
                f"[ERROR] Unsupported results format in {filepath}: "
                f"{type(results).__name__} - skipping row"
            )
            continue
        if surfaces:
            rows.append((stem, surfaces))
        else:
            print(f"[WARNING] No plottable data in {filepath} - skipping row")
    return rows


def _paper_palette(n: int) -> list:
    """Return ``n`` visually-distinct RGBA colors. Uses a curated qualitative
    list first (tab10 + tab20b, de-duplicated) and, if more are needed, extends
    with evenly-spaced HSV samples so that no two surfaces anywhere in the
    figure share a color (requirement #6)."""
    from matplotlib.colors import to_rgba
    import colorsys

    curated = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20b").colors)
    seen: set = set()
    palette: list = []
    for c in curated:
        rgba = to_rgba(c)
        if rgba not in seen:
            seen.add(rgba)
            palette.append(rgba)
    if n > len(palette):
        extra = n - len(palette)
        for i in range(extra):
            h = i / max(1, extra)
            palette.append(to_rgba(colorsys.hsv_to_rgb(h, 0.65, 0.85)))
    return palette[:n]


def assign_paper_colors(
    rows: list[tuple[str, list[tuple[str, dict]]]],
) -> list[list]:
    """Assign a globally-unique color to every surface across all rows.

    Returns ``colors_by_row`` so that ``colors_by_row[r][s]`` is the RGBA for
    surface ``s`` of row ``r``. The same color is reused for that surface in all
    three metric columns, and no color repeats anywhere in the figure."""
    total = sum(len(surfaces) for _stem, surfaces in rows)
    palette = _paper_palette(total)
    colors_by_row: list[list] = []
    idx = 0
    for _stem, surfaces in rows:
        row_colors = []
        for _ in surfaces:
            row_colors.append(palette[idx])
            idx += 1
        colors_by_row.append(row_colors)
    return colors_by_row


def _paper_zmax(
    rows: list[tuple[str, list[tuple[str, dict]]]],
    eps_e1: list[float],
    eps_e2: list[float],
    *,
    include_std: bool,
) -> dict[str, float]:
    """Per-metric global z-max across every surface of every row (requirement
    #5). When ``include_std`` is True the whisker tops (``mean + std``) are
    included so error bars are never clipped. Keyed by the metric's mean key."""
    zmax: dict[str, float] = {}
    for mean_key, std_key, _label in _PAPER_METRICS:
        gmax = 0.0
        for _stem, surfaces in rows:
            for _slabel, agg in surfaces:
                for e1 in eps_e1:
                    for e2 in eps_e2:
                        cell = agg.get((e1, e2), {})
                        m = float(cell.get(mean_key, 0.0))
                        s = float(cell.get(std_key, 0.0)) if include_std else 0.0
                        gmax = max(gmax, m + s)
        zmax[mean_key] = max(0.1, gmax * 1.05)
    return zmax


def plot_paper_grid(
    rows: list[tuple[str, list[tuple[str, dict]]]],
    *,
    eps_values_e1: list[float],
    eps_values_e2: list[float],
    out_stem: str,
    draw_std: bool = True,
    row_labels: list[str] | None = None,
) -> None:
    """Render the paper-format comparison figure.

    Layout: one pkl per row, three metric columns (throughput / mean delay /
    max delay). Header-free (no per-subplot titles, no suptitle, requirement
    #2/#4 -- the metric name lives on the z-axis / side), no capacity surface
    (requirement #3), opaque mesh surfaces, optional std whiskers, one legend
    per row on the throughput (col-0) subplot (requirement #6), and a shared
    per-metric z-scale across all rows (requirement #5). Saves both PNG and a
    vector PDF."""
    n_rows = len(rows)
    n_e1 = len(eps_values_e1)
    n_e2 = len(eps_values_e2)
    EPS1, EPS2 = np.meshgrid(eps_values_e1, eps_values_e2)

    colors_by_row = assign_paper_colors(rows)
    zmax_by_metric = _paper_zmax(
        rows, eps_values_e1, eps_values_e2, include_std=draw_std
    )

    fig = plt.figure(figsize=(20, 6 * n_rows))
    axes_grid: list[list] = []
    for r, (_stem, surfaces) in enumerate(rows):
        row_colors = colors_by_row[r]
        row_axes = []
        for c, (mean_key, std_key, side_label) in enumerate(_PAPER_METRICS):
            ax = fig.add_subplot(n_rows, 3, r * 3 + c + 1, projection="3d")
            row_axes.append(ax)
            legend_patches = []
            for s_idx, (surf_label, agg) in enumerate(surfaces):
                color = row_colors[s_idx]

                mean_grid = np.zeros((n_e1, n_e2))
                for i, e1 in enumerate(eps_values_e1):
                    for j, e2 in enumerate(eps_values_e2):
                        mean_grid[i, j] = agg.get((e1, e2), {}).get(mean_key, 0.0)

                # Opaque surface with a light mesh so overlapping surfaces still
                # read as solid, paper-style panels.
                ax.plot_surface(
                    EPS1,
                    EPS2,
                    mean_grid.T,
                    color=color,
                    edgecolor="k",
                    linewidth=0.2,
                    alpha=1.0,
                )

                if draw_std:
                    for i, e1 in enumerate(eps_values_e1):
                        for j, e2 in enumerate(eps_values_e2):
                            s = agg.get((e1, e2), {}).get(std_key, 0.0)
                            if s <= 0:
                                continue
                            m = mean_grid[i, j]
                            ax.plot(
                                [e1, e1],
                                [e2, e2],
                                [m - s, m + s],
                                color="black",
                                linewidth=0.7,
                                alpha=0.5,
                            )

                legend_patches.append(Patch(color=color, label=surf_label))

            ax.set_xlabel("ε₁")
            ax.set_ylabel("ε₂")
            ax.set_zlabel(side_label)
            ax.set_zlim(0, zmax_by_metric[mean_key])
            # Match the per-k / SR orientation: shallow pitch, both eps axes
            # inverted so the low-eps corner faces the viewer.
            ax.view_init(elev=10, azim=-45)
            ax.invert_xaxis()
            ax.invert_yaxis()
            # One legend per pkl-row, nestled inside the upper-left of the
            # throughput (col-0) plot like paper Fig. 19 (framed box sitting on
            # the plot rather than floating above it). bbox_to_anchor pulls it
            # down into the 3D cube's empty upper-left corner.
            if c == 0:
                leg = ax.legend(
                    handles=legend_patches,
                    fontsize=9,
                    loc="upper left",
                    bbox_to_anchor=(0.04, 0.72),
                    framealpha=1.0,
                    edgecolor="0.3",
                    fancybox=False,
                    borderpad=0.6,
                )
                # In 3D, plot_surface collections receive dynamic (depth-based)
                # zorders on every draw and can paint over the legend, leaving it
                # hidden behind the surfaces. Forcing a very high zorder keeps the
                # (opaque) legend box drawn on top of all surfaces.
                leg.set_zorder(10000)

        axes_grid.append(row_axes)

    plt.tight_layout()

    # Per-row protocol headers: a bold title centered above each row's three
    # panels (placed after tight_layout so axes positions are final).
    if row_labels:
        for r, row_axes in enumerate(axes_grid):
            if r >= len(row_labels) or not row_labels[r]:
                continue
            positions = [a.get_position() for a in row_axes]
            x_center = (
                min(p.x0 for p in positions) + max(p.x1 for p in positions)
            ) / 2.0
            y_top = max(p.y1 for p in positions)
            fig.text(
                x_center,
                min(0.995, y_top + 0.005),
                row_labels[r],
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
            )
    png_path = f"{out_stem}.png"
    pdf_path = f"{out_stem}.pdf"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"[OK] Paper plot saved to: {png_path}")
    print(f"[OK] Paper plot saved to: {pdf_path}")
    plt.show()


def plot_stats_comparison(datasets: list[tuple[str, list]]):
    """
    Plot comparison of multiple protocols on the same 3D graphs.
    
    Args:
        datasets: List of tuples (label, results) where results is list of (eps1, eps2, SimulationStats)
    """
    # Define colors and styles for different protocols
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    colormaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'copper', 'RdPu', 'Greys']
    
    # Get unique eps1 and eps2 values from first dataset (assume all use same grid)
    aggregated_first = aggregate_results(datasets[0][1])
    eps_pairs = sorted(aggregated_first.keys())
    unique_eps1 = sorted(list(set([k[0] for k in eps_pairs])))
    unique_eps2 = sorted(list(set([k[1] for k in eps_pairs])))
    
    n_eps1 = len(unique_eps1)
    n_eps2 = len(unique_eps2)
    
    # Create meshgrid for plotting
    EPS1, EPS2 = np.meshgrid(unique_eps1, unique_eps2)
    
    # Add channel capacity calculation
    capacity_grid = np.zeros((n_eps1, n_eps2))
    for i, eps1 in enumerate(unique_eps1):
        for j, eps2 in enumerate(unique_eps2):
            # Capacity = sum of (1 - eps) for all 4 channels (assuming eps3=0.2, eps4=0.8)
            capacity_grid[i, j] = (1 - eps1) + (1 - eps2) + (1 - 0.2) + (1 - 0.8)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))
    
    # ========================================
    # Plot 1: Normalized Throughput Comparison
    # ========================================
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot capacity surface first (as reference)
    ax1.plot_surface(EPS2, EPS1, capacity_grid.T, color='red', 
                    edgecolor='none', alpha=0.2, label='Capacity')
    
    # Plot each protocol's throughput
    for idx, (label, results) in enumerate(datasets):
        aggregated = aggregate_results(results)
        
        throughput_mean_grid = np.zeros((n_eps1, n_eps2))
        throughput_std_grid = np.zeros((n_eps1, n_eps2))
        
        for (eps1, eps2), agg_data in aggregated.items():
            i = unique_eps1.index(eps1)
            j = unique_eps2.index(eps2)
            throughput_mean_grid[i, j] = agg_data['throughput_mean']
            throughput_std_grid[i, j] = agg_data['throughput_std']
        
        # Plot surface with protocol-specific colormap
        cmap = colormaps[idx % len(colormaps)]
        surf = ax1.plot_surface(EPS2, EPS1, throughput_mean_grid.T, cmap=cmap, 
                                edgecolor='none', alpha=0.6)
        
        # Add error bars
        color = colors[idx % len(colors)]
        for i, eps1 in enumerate(unique_eps1):
            for j, eps2 in enumerate(unique_eps2):
                mean_val = throughput_mean_grid[i, j]
                std_val = throughput_std_grid[i, j]
                ax1.plot([eps2, eps2], [eps1, eps1], 
                        [mean_val - std_val, mean_val + std_val],
                        color=color, linewidth=1.0, alpha=0.7)
    
    ax1.set_xlabel('Epsilon 2 (Path 1)', fontsize=10)
    ax1.set_ylabel('Epsilon 1 (Path 0)', fontsize=10)
    ax1.set_zlabel('Normalized Throughput', fontsize=10)
    ax1.set_zlim(0, 3)  # Set z-axis limits for throughput
    ax1.set_title('Normalized Throughput Comparison', fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    ax1.invert_yaxis()
    
    # ========================================
    # Plot 2: Mean In-Order Delay Comparison
    # ========================================
    ax2 = fig.add_subplot(132, projection='3d')
    
    for idx, (label, results) in enumerate(datasets):
        aggregated = aggregate_results(results)
        
        delay_mean_grid = np.zeros((n_eps1, n_eps2))
        delay_mean_std_grid = np.zeros((n_eps1, n_eps2))
        
        for (eps1, eps2), agg_data in aggregated.items():
            i = unique_eps1.index(eps1)
            j = unique_eps2.index(eps2)
            delay_mean_grid[i, j] = agg_data['delay_mean_mean']
            delay_mean_std_grid[i, j] = agg_data['delay_mean_std']
        
        # Plot surface
        cmap = colormaps[idx % len(colormaps)]
        surf = ax2.plot_surface(EPS2, EPS1, delay_mean_grid.T, cmap=cmap,
                                edgecolor='none', alpha=0.6)
        
        # Add error bars
        color = colors[idx % len(colors)]
        for i, eps1 in enumerate(unique_eps1):
            for j, eps2 in enumerate(unique_eps2):
                mean_val = delay_mean_grid[i, j]
                std_val = delay_mean_std_grid[i, j]
                ax2.plot([eps2, eps2], [eps1, eps1], 
                        [mean_val - std_val, mean_val + std_val],
                        color=color, linewidth=1.0, alpha=0.7)
    
    ax2.set_xlabel('Epsilon 2 (Path 1)', fontsize=10)
    ax2.set_ylabel('Epsilon 1 (Path 0)', fontsize=10)
    ax2.set_zlabel('Mean In-Order Delay', fontsize=10)
    ax2.set_zlim(0, 600)  # Set z-axis limits for mean delay
    ax2.set_title('Mean In-Order Delay Comparison', fontsize=12, fontweight='bold')
    ax2.view_init(elev=20, azim=45)
    ax2.invert_yaxis()
    
    # ========================================
    # Plot 3: Max In-Order Delay Comparison
    # ========================================
    ax3 = fig.add_subplot(133, projection='3d')
    
    for idx, (label, results) in enumerate(datasets):
        aggregated = aggregate_results(results)
        
        delay_max_grid = np.zeros((n_eps1, n_eps2))
        delay_max_std_grid = np.zeros((n_eps1, n_eps2))
        
        for (eps1, eps2), agg_data in aggregated.items():
            i = unique_eps1.index(eps1)
            j = unique_eps2.index(eps2)
            delay_max_grid[i, j] = agg_data['delay_max_mean']
            delay_max_std_grid[i, j] = agg_data['delay_max_std']
        
        # Plot surface
        cmap = colormaps[idx % len(colormaps)]
        surf = ax3.plot_surface(EPS2, EPS1, delay_max_grid.T, cmap=cmap,
                                edgecolor='none', alpha=0.6)
        
        # Add error bars
        color = colors[idx % len(colors)]
        for i, eps1 in enumerate(unique_eps1):
            for j, eps2 in enumerate(unique_eps2):
                mean_val = delay_max_grid[i, j]
                std_val = delay_max_std_grid[i, j]
                ax3.plot([eps2, eps2], [eps1, eps1], 
                        [mean_val - std_val, mean_val + std_val],
                        color=color, linewidth=1.0, alpha=0.7)
    
    ax3.set_xlabel('Epsilon 2 (Path 1)', fontsize=10)
    ax3.set_ylabel('Epsilon 1 (Path 0)', fontsize=10)
    ax3.set_zlabel('Max In-Order Delay', fontsize=10)
    ax3.set_zlim(0, 600)  # Set z-axis limits for max delay
    ax3.set_title('Max In-Order Delay Comparison', fontsize=12, fontweight='bold')
    ax3.view_init(elev=20, azim=45)
    ax3.invert_yaxis()
    
    # Add legend with protocol labels
    legend_text = '\n'.join([f'{colors[i % len(colors)]}: {label}' 
                            for i, (label, _) in enumerate(datasets)])
    
    ## Add overall title with protocol names
    # protocols_str = ' vs '.join([label for label, _ in datasets])
    # fig.suptitle(f'Protocol Comparison: {protocols_str}\n(Path 2: eps=0.2, Path 3: eps=0.8)',
    #              fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('protocol_comparison_3d.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison plots saved to: protocol_comparison_3d.png")
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "Plot Saved Results")
    print("="*70)
    
    # Parse CLI: -paper/--paper flag, optional -o/--out PATH, and the file list.
    file_args, paper_mode, out_stem, row_labels_arg = _parse_cli_args(sys.argv[1:])
    if file_args:
        results_files = file_args
    else:
        current_dir = Path('.')
        mp_sim_results = current_dir / 'simulation_results.pkl'
        sr_sim_results = current_dir / 'sr_arq' / 'results_sr' / 'sr_simulation_results.pkl'
        results_files = [mp_sim_results, sr_sim_results]
        # results_files = [sr_sim_results]
    if paper_mode:
        print("[paper] Paper-format mode enabled "
              "(stacked rows, shared per-metric scale, PNG + PDF).")
    
    # Check if files exist
    valid_files = []
    for f in results_files:
        if os.path.exists(f):
            valid_files.append(f)
        else:
            print(f"[WARNING] File not found: {f} - skipping")
    
    if not valid_files:
        print("[ERROR] No valid result files found!")
        sys.exit(1)

    # Paper-format mode: one pkl per row, 3 metric columns, shared per-metric
    # z-scale, globally-unique surface colors, one legend per row, no headers /
    # capacity. Handled here before the standard routing below.
    if paper_mode:
        print(f"\nBuilding paper-format figure from {len(valid_files)} file(s)...")
        print("="*70)
        rows = build_paper_rows(valid_files)
        if not rows:
            print("[ERROR] No plottable data found for paper mode.")
            sys.exit(1)
        eps1, eps2 = _eps_grid_from_aggs(
            [agg for _stem, surfaces in rows for _lbl, agg in surfaces]
        )
        # Resolve one protocol header per row: use --row-labels when given
        # (in file order), otherwise auto-detect from the filename.
        row_headers = [
            row_labels_arg[idx]
            if idx < len(row_labels_arg) and row_labels_arg[idx]
            else _auto_protocol_label(stem)
            for idx, (stem, _surfaces) in enumerate(rows)
        ]
        plot_paper_grid(
            rows,
            eps_values_e1=eps1,
            eps_values_e2=eps2,
            out_stem=out_stem,
            draw_std=True,
            row_labels=row_headers,
        )
        print(f"\n{'='*70}")
        print("Plots saved successfully!")
        print("="*70)
        sys.exit(0)

    # Load results from all files; auto-route per-k / multi-series / flat-list
    flat_datasets: list[tuple[str, list]] = []
    per_k_datasets: list[tuple[str, dict, str]] = []
    print(f"\nLoading {len(valid_files)} result file(s)...")
    print("="*70)
    
    for filepath in valid_files:
        # Extract label from filename (remove path and extension)
        label = os.path.splitext(os.path.basename(filepath))[0]
        
        results = load_results_any(filepath)
        if is_per_k_results(results):
            per_k_datasets.append((label, results, str(filepath)))
        elif is_multi_series_results(results):
            # Expand {setting: [(e1,e2,stats), ...]} into one flat series each
            for key, series_results in results.items():
                if not series_results:
                    continue
                flat_datasets.append((series_label(key, label), series_results))
        elif isinstance(results, list):
            flat_datasets.append((label, results))
        else:
            print(f"[ERROR] Unsupported results format in {filepath}: {type(results).__name__}")
            sys.exit(1)
    
    # Plot
    print(f"\n{'='*70}")
    print("Generating plots...")
    print("="*70)
    
    # Mixed flat + per-k, OR multiple per-k pickles -> unified overlay path
    if (flat_datasets and per_k_datasets) or len(per_k_datasets) > 1:
        print(
            f"Mixed/multiple datasets -> overlay plot "
            f"({len(flat_datasets)} flat, {len(per_k_datasets)} per-k)"
        )
        series, capacity_surfaces = build_overlay_series_and_capacities(
            flat_datasets, per_k_datasets
        )
        if not series:
            print("[ERROR] No data to plot (all per-k datasets were empty).")
            sys.exit(1)
        eps1, eps2 = _eps_grid_from_aggs([agg for _label, agg in series])
        labels_blob = " + ".join(
            [lbl for lbl, _ in flat_datasets] + [lbl for lbl, _, _ in per_k_datasets]
        )
        plot_overlay_surfaces(
            series,
            eps_values_e1=eps1,
            eps_values_e2=eps2,
            title_suffix=f"Overlay: {labels_blob}",
            plot_path="combined_overlay.png",
            capacity_surfaces=capacity_surfaces,
        )
    elif len(per_k_datasets) == 1:
        label, results, filepath = per_k_datasets[0]
        s_label, s_fmt = series_kind_for_file(filepath)
        noun = "alpha" if s_label == "α" else "k"
        print(f"Single per-{noun} dataset - using overlaid surface plot")
        plot_stem = os.path.splitext(os.path.basename(filepath))[0]
        plot_per_k_stats(
            results,
            title_suffix=f"{label} (1 surface per {s_label})",
            plot_path=f"{plot_stem}.png",
            series_label=s_label,
            series_fmt=s_fmt,
        )
    elif len(flat_datasets) >= 1:
        # One or more flat / multi-series datasets - SR-style view
        # (eps1 left, eps2 right, Z right; both eps increase toward the back).
        # Always use plot_compare — even for a single series — so single-key
        # pickles like sr_arq_mp_results.pkl don't fall back to plot_stats'
        # old azim=45 orientation.
        n = len(flat_datasets)
        print(
            f"{'Single protocol' if n == 1 else f'Comparing {n} protocols/series'}"
            " - using SR-style plot"
        )
        colors = [
            "tab:purple", "tab:red", "tab:blue", "tab:green",
            "tab:orange", "tab:brown", "tab:pink", "tab:gray",
        ]
        series = []
        eps1_vals: set[float] = set()
        eps2_vals: set[float] = set()
        for idx, (label, results) in enumerate(flat_datasets):
            agg = _snap_agg_keys(sr_aggregate(results))
            series.append((label, agg, colors[idx % len(colors)]))
            for e1, e2 in agg:
                eps1_vals.add(e1)
                eps2_vals.add(e2)
        e1_sorted = sorted(eps1_vals)
        e2_sorted = sorted(eps2_vals)
        labels_blob = " vs ".join(lbl for lbl, _ in flat_datasets)
        plot_stem = "protocol_comparison_3d"
        if len(valid_files) == 1:
            plot_stem = os.path.splitext(os.path.basename(valid_files[0]))[0]
        sr_plot_compare(
            series,
            e1_sorted,
            e2_sorted,
            title_suffix=f"Comparison: {labels_blob}",
            plot_path=f"{plot_stem}.png",
            std_for=series[-1][0] if series else None,
        )
    
    print(f"\n{'='*70}")
    print("Plots saved successfully!")
    print("="*70)

