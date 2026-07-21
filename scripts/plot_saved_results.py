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
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "jamming_simulation"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import numpy as np
import matplotlib.pyplot as plt
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
    """True when obj is {k: [(e1, e2, stats), ...]} from jam sweep_eps_grid_per_k.

    Tolerates empty value lists (a partial pickle written by the incremental
    per-k save during a run that hasn't yet started one or more k values).
    Requires at least one k to have non-empty data so the format can be
    distinguished from arbitrary {int: list} dicts.
    """
    if not isinstance(obj, dict) or not obj:
        return False
    saw_any_data = False
    for key, value in obj.items():
        if not isinstance(key, int) or not isinstance(value, list):
            return False
        if not value:
            continue  # partial pickle: this k hasn't started yet
        saw_any_data = True
        first = value[0]
        if not isinstance(first, tuple) or len(first) != 3:
            return False
    return saw_any_data


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
        empty_ks = sorted(set(results) - set(non_empty))
        total = sum(len(v) for v in non_empty.values())
        print(
            f"     Per-k format: {len(results)} k value(s) "
            f"({len(non_empty)} non-empty, {len(empty_ks)} empty), "
            f"{total} total data points"
        )
        if empty_ks:
            print(f"     Empty k value(s) (skipped in plot): {empty_ks}")
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
) -> None:
    aggregated_per_k = aggregate_per_k_results(per_k_results)
    if not aggregated_per_k:
        print("[ERROR] No completed k values to plot.")
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
    for label, per_k_results, _filepath in per_k_datasets:
        for k in sorted(per_k_results.keys()):
            if not per_k_results[k]:
                continue
            agg = _snap_agg_keys(aggregate_results(per_k_results[k]))
            series.append((f"{label} k={k}", agg))

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
    
    # Get filenames from command line or use default
    if len(sys.argv) > 1:
        results_files = sys.argv[1:]
    else:
        current_dir = Path('.')
        mp_sim_results = current_dir / 'simulation_results.pkl'
        sr_sim_results = current_dir / 'sr_arq' / 'results_sr' / 'sr_simulation_results.pkl'
        results_files = [mp_sim_results, sr_sim_results]
        # results_files = [sr_sim_results]
    
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
        print("Single per-k dataset - using overlaid surface plot")
        plot_stem = os.path.splitext(os.path.basename(filepath))[0]
        plot_per_k_stats(
            results,
            title_suffix=f"{label} (1 surface per k)",
            plot_path=f"{plot_stem}.png",
        )
    elif len(flat_datasets) == 1:
        # Single dataset - use original plotting function
        print("Single protocol - using standard plot")
        plot_stats(flat_datasets[0][1])
    else:
        # Multiple flat / multi-series datasets - SR-style overlay comparison
        print(f"Comparing {len(flat_datasets)} protocols/series")
        colors = [
            "tab:red", "tab:blue", "tab:green", "tab:orange",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
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

