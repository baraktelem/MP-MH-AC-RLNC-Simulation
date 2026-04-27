"""
Multi-hop multipath AC-RLNC simulation driver (MpMhNetwork only).

Mirrors mp_simulation.py (pickle, aggregation, 3D plots) for Cohen et al. MP-MH
setting: H=3, P=4, RTT=12, th=0, ō=2k with k=P(RT T−1), 200 packets, 150 sweeps.

Usage:
    conda activate mp_mh_ac_rlnc   # env where NetworkX is installed (conda-forge)
    python mh_mp_simulation.py

    Or: conda run -n mp_mh_ac_rlnc python mh_mp_simulation.py

    Set LOAD_EXISTING = True to replot from mh_mp_simulation_results.pkl
    without re-running simulations.

    Set DEBUG = True in __main__ to tee stdout and stderr to a log file (default
    mp_mh_sim.log, or DEBUG_LOG_FILE). Uncaught exceptions and tracebacks go to
    stderr, so they are captured in the log as well.

    MpMhNetwork(debug=DEBUG) controls protocol verbosity vs Network suppression.
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from mh_epsilon_matrix import (
    DEFAULT_NUM_HOPS,
    DEFAULT_NUM_PATHS,
    build_path_epsilons,
    print_article_epsilon_matrix,
    validate_article_matrix,
    article_matrix_for_hops,
)
from mh_min_cut_capacity import min_cut_capacity_for_epsilons
from Network import MpMhNetwork, SimulationStats


def effective_num_hops(num_hops: int) -> int:
    if num_hops in (0, 1):
        return 1
    return num_hops


def save_results(
    results: list[tuple[float, float, SimulationStats]], filename: str | None = None
) -> str:
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mp_mh_simulation_results_{timestamp}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"[OK] Results saved to: {filename}")
    return filename


def load_results(filename: str) -> list[tuple[float, float, SimulationStats]]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file not found: {filename}")
    with open(filename, "rb") as f:
        results = pickle.load(f)
    print(f"[OK] Results loaded from: {filename}")
    print(f"     Total data points: {len(results)}")
    return results


def aggregate_results(
    results: list[tuple[float, float, SimulationStats]],
) -> dict[tuple[float, float], dict]:
    grouped: defaultdict = defaultdict(
        lambda: {"throughput": [], "delay_mean": [], "delay_max": []}
    )
    for eps1, eps2, stats in results:
        key = (eps1, eps2)
        grouped[key]["throughput"].append(stats.normalized_throughput)
        grouped[key]["delay_mean"].append(stats.inorder_delay_mean)
        grouped[key]["delay_max"].append(stats.inorder_delay_max)

    aggregated = {}
    for key, values in grouped.items():
        aggregated[key] = {
            "throughput_mean": np.mean(values["throughput"]),
            "throughput_std": np.std(values["throughput"]),
            "delay_mean_mean": np.mean(values["delay_mean"]),
            "delay_mean_std": np.std(values["delay_mean"]),
            "delay_max_mean": np.mean(values["delay_max"]),
            "delay_max_std": np.std(values["delay_max"]),
        }
    return aggregated


def plot_stats(
    results: list[tuple[float, float, SimulationStats]],
    num_hops_eff: int,
    plot_path: str = "mp_mh_performance_3d.png",
) -> None:
    aggregated = aggregate_results(results)

    all_throughput_stds = [v["throughput_std"] for v in aggregated.values()]
    all_delay_mean_stds = [v["delay_mean_std"] for v in aggregated.values()]
    all_delay_max_stds = [v["delay_max_std"] for v in aggregated.values()]

    print("\nStandard Deviation Statistics:")
    print(
        f"  Throughput std - min: {np.min(all_throughput_stds):.6f}, "
        f"max: {np.max(all_throughput_stds):.6f}, "
        f"mean: {np.mean(all_throughput_stds):.6f}"
    )
    print(
        f"  Delay mean std - min: {np.min(all_delay_mean_stds):.6f}, "
        f"max: {np.max(all_delay_mean_stds):.6f}, "
        f"mean: {np.mean(all_delay_mean_stds):.6f}"
    )
    print(
        f"  Delay max std  - min: {np.min(all_delay_max_stds):.6f}, "
        f"max: {np.max(all_delay_max_stds):.6f}, "
        f"mean: {np.mean(all_delay_max_stds):.6f}"
    )
    print()

    eps_pairs = sorted(aggregated.keys())
    unique_eps1 = sorted({k[0] for k in eps_pairs})
    unique_eps2 = sorted({k[1] for k in eps_pairs})
    n_eps1 = len(unique_eps1)
    n_eps2 = len(unique_eps2)

    throughput_mean_grid = np.zeros((n_eps1, n_eps2))
    throughput_std_grid = np.zeros((n_eps1, n_eps2))
    delay_mean_mean_grid = np.zeros((n_eps1, n_eps2))
    delay_mean_std_grid = np.zeros((n_eps1, n_eps2))
    delay_max_mean_grid = np.zeros((n_eps1, n_eps2))
    delay_max_std_grid = np.zeros((n_eps1, n_eps2))

    for (eps1, eps2), agg_data in aggregated.items():
        i = unique_eps1.index(eps1)
        j = unique_eps2.index(eps2)
        throughput_mean_grid[i, j] = agg_data["throughput_mean"]
        throughput_std_grid[i, j] = agg_data["throughput_std"]
        delay_mean_mean_grid[i, j] = agg_data["delay_mean_mean"]
        delay_mean_std_grid[i, j] = agg_data["delay_mean_std"]
        delay_max_mean_grid[i, j] = agg_data["delay_max_mean"]
        delay_max_std_grid[i, j] = agg_data["delay_max_std"]

    EPS1, EPS2 = np.meshgrid(unique_eps1, unique_eps2)

    capacity_grid = np.zeros((n_eps1, n_eps2))
    for i, eps1 in enumerate(unique_eps1):
        for j, eps2 in enumerate(unique_eps2):
            capacity_grid[i, j] = min_cut_capacity_for_epsilons(
                eps1, eps2, num_hops_eff
            )

    zmax_tp = max(3.0, float(np.max(throughput_mean_grid)), float(np.max(capacity_grid)))
    zmax_delay = max(600.0, float(np.max(delay_mean_mean_grid)), float(np.max(delay_max_mean_grid)))

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(
        EPS1,
        EPS2,
        throughput_mean_grid.T,
        cmap="viridis",
        edgecolor="none",
        alpha=0.7,
    )
    ax1.plot_surface(
        EPS1, EPS2, capacity_grid.T, color="red", edgecolor="none", alpha=0.3
    )
    for i, eps1 in enumerate(unique_eps1):
        for j, eps2 in enumerate(unique_eps2):
            mean_val = throughput_mean_grid[i, j]
            std_val = throughput_std_grid[i, j]
            ax1.plot(
                [eps1, eps1],
                [eps2, eps2],
                [mean_val - std_val, mean_val + std_val],
                color="black",
                linewidth=1.5,
                alpha=0.8,
            )
    ax1.set_xlabel("ε₁ (paper Path 1, Hop 1 & …)", fontsize=10)
    ax1.set_ylabel("ε₂ (paper Paths 3–4, …)", fontsize=10)
    ax1.set_zlabel("Normalized Throughput", fontsize=10)
    ax1.set_zlim(0, zmax_tp)
    ax1.set_title("Normalized Throughput vs ε₁, ε₂", fontsize=12, fontweight="bold")
    ax1.view_init(elev=20, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(
        EPS1,
        EPS2,
        delay_mean_mean_grid.T,
        cmap="plasma",
        edgecolor="none",
        alpha=0.7,
    )
    for i, eps1 in enumerate(unique_eps1):
        for j, eps2 in enumerate(unique_eps2):
            mean_val = delay_mean_mean_grid[i, j]
            std_val = delay_mean_std_grid[i, j]
            ax2.plot(
                [eps1, eps1],
                [eps2, eps2],
                [mean_val - std_val, mean_val + std_val],
                color="black",
                linewidth=1.5,
                alpha=0.8,
            )
    ax2.set_xlabel("ε₁", fontsize=10)
    ax2.set_ylabel("ε₂", fontsize=10)
    ax2.set_zlabel("Mean In-Order Delay", fontsize=10)
    ax2.set_zlim(0, zmax_delay)
    ax2.set_title("Mean In-Order Delay vs ε₁, ε₂", fontsize=12, fontweight="bold")
    ax2.view_init(elev=20, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    ax3 = fig.add_subplot(133, projection="3d")
    surf3 = ax3.plot_surface(
        EPS1,
        EPS2,
        delay_max_mean_grid.T,
        cmap="inferno",
        edgecolor="none",
        alpha=0.7,
    )
    for i, eps1 in enumerate(unique_eps1):
        for j, eps2 in enumerate(unique_eps2):
            mean_val = delay_max_mean_grid[i, j]
            std_val = delay_max_std_grid[i, j]
            ax3.plot(
                [eps1, eps1],
                [eps2, eps2],
                [mean_val - std_val, mean_val + std_val],
                color="black",
                linewidth=1.5,
                alpha=0.8,
            )
    ax3.set_xlabel("ε₁", fontsize=10)
    ax3.set_ylabel("ε₂", fontsize=10)
    ax3.set_zlabel("Max In-Order Delay", fontsize=10)
    ax3.set_zlim(0, zmax_delay)
    ax3.set_title("Max In-Order Delay vs ε₁, ε₂", fontsize=12, fontweight="bold")
    ax3.view_init(elev=20, azim=45)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    fig.suptitle(
        f"MpMhNetwork (H={num_hops_eff}, P={DEFAULT_NUM_PATHS}) — Cohen et al. ε template\n"
        "Red surface: min-cut / max-flow (NetworkX, layered BEC); averaged over iterations",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[OK] 3D plots saved to: {plot_path}")
    plt.show()


class _StdoutTee:
    """Write to multiple text streams (console + log file)."""

    __slots__ = ("_streams",)

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _run_main(*, debug: bool = False) -> None:
    """
    debug: When True (script DEBUG), stdout/stderr are teed to the log; we pass
    MpMhNetwork(debug=True) so Sender/Node/Receiver prints during run_step are
    not redirected to devnull.
    """
    print("=" * 70)
    print(" " * 12 + "MP-MH AC-RLNC — MpMhNetwork simulation")
    print("=" * 70)

    eps_values = list(np.arange(0.1, 0.9, 0.1))
    NUM_PATHS = DEFAULT_NUM_PATHS
    # NUM_HOPS = DEFAULT_NUM_HOPS
    NUM_HOPS = 1
    num_hops_eff = effective_num_hops(NUM_HOPS)

    # RTT = 12
    RTT = 20
    PROP_DELAY = RTT // 2
    THRESHOLD = 0.0
    k_mp = NUM_PATHS * (RTT - 1)
    O_BAR = 2 * NUM_PATHS * (RTT - 1)
    NUM_PACKETS_TO_SEND = 200
    MAX_ITERATIONS = None
    NUM_ITERATIONS = 150
    LOAD_EXISTING = False
    RESULTS_FILE = "mp_mh_simulation_results.pkl"
    PLOT_FILE = "mp_mh_performance_3d.png"

    article_preview = article_matrix_for_hops(0.1, 0.2, num_hops_eff)
    validate_article_matrix(article_preview, NUM_PATHS, num_hops_eff)

    print("\nSimulation parameters:")
    print(f"  - RTT (slots): {RTT}, prop_delay: {PROP_DELAY}")
    print(f"  - Threshold: {THRESHOLD}")
    print(f"  - k = P(RTT−1) = {k_mp}, ō = 2k = {O_BAR} (paper)")
    print(f"  - Paths P = {NUM_PATHS}, hops H = {num_hops_eff} (NUM_HOPS config = {NUM_HOPS})")
    print(f"  - Packets per run: {NUM_PACKETS_TO_SEND}, max_iterations: {MAX_ITERATIONS}")
    print(f"  - Outer iterations: {NUM_ITERATIONS}")
    print(
        f"  - DEBUG: {debug} (stdout tee + MpMhNetwork(debug={debug}) "
        f"so Sender/Node/Receiver prints reach the log)"
    )
    print_article_epsilon_matrix(0.1, 0.2, NUM_PATHS, num_hops_eff)

    if LOAD_EXISTING and os.path.exists(RESULTS_FILE):
        print(f"\n{'=' * 70}\nLoading existing results...\n{'=' * 70}")
        results = load_results(RESULTS_FILE)
    else:
        print(f"\n{'=' * 70}\nRunning new simulations...\n{'=' * 70}")
        results = []
        total_per_iter = len(eps_values) ** 2
        total_sims = total_per_iter * NUM_ITERATIONS
        sim_count = 0
        print(
            f"\nTotal simulations: {total_sims} "
            f"({NUM_ITERATIONS} × {total_per_iter} (ε₁,ε₂) pairs)\n{'=' * 70}\n"
        )

        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"\n{'=' * 70}\nIteration {iteration}/{NUM_ITERATIONS}\n{'=' * 70}\n")
            for eps1 in eps_values:
                for eps2 in eps_values:
                    sim_count += 1
                    path_eps = build_path_epsilons(eps1, eps2, num_hops_eff)
                    cap = min_cut_capacity_for_epsilons(eps1, eps2, num_hops_eff)
                    print(
                        f"[{sim_count}/{total_sims}] iter {iteration}/{NUM_ITERATIONS}: "
                        f"ε₁={eps1:.1f} ε₂={eps2:.1f} | min-cut ref={cap:.4f}"
                    )
                    network = MpMhNetwork(
                        path_epsilons=path_eps,
                        initial_epsilon=0.5,
                        max_iterations=MAX_ITERATIONS,
                        num_packets_to_send=NUM_PACKETS_TO_SEND,
                        max_allowed_overlap=O_BAR,
                        num_paths=NUM_PATHS,
                        prop_delay=PROP_DELAY,
                        threshold=THRESHOLD,
                        num_hops=num_hops_eff,
                        debug=debug,
                    )
                    network.run_sim()
                    stats = network.get_simulation_stats()
                    results.append((eps1, eps2, stats))
                    print(
                        f"  → throughput: {stats.normalized_throughput:.4f}, "
                        f"mean delay: {stats.inorder_delay_mean:.2f}, "
                        f"max delay: {stats.inorder_delay_max}"
                    )

        print(f"\n{'=' * 70}\nAll {total_sims} simulations completed.\n{'=' * 70}")
        save_results(results, RESULTS_FILE)

    print(f"\n{'=' * 70}\nAggregating and plotting...\n{'=' * 70}")
    plot_stats(results, num_hops_eff, plot_path=PLOT_FILE)
    print(f"\n{'=' * 70}\nDone.\n{'=' * 70}")


if __name__ == "__main__":
    # When True, tee stdout and stderr to the log so prints and crash tracebacks are saved.
    DEBUG = False
    DEBUG_LOG_FILE: str | None = None  # default file name below if None

    _log_fp = None
    _orig_stdout = None
    _orig_stderr = None
    if DEBUG:
        log_path = DEBUG_LOG_FILE or "mp_mh_sim.log"
        _log_fp = open(log_path, "w", encoding="utf-8")
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr
        sys.stdout = _StdoutTee(_orig_stdout, _log_fp)
        sys.stderr = _StdoutTee(_orig_stderr, _log_fp)
        print(f"[DEBUG] Logging stdout/stderr to: {os.path.abspath(log_path)}")

    try:
        _run_main(debug=DEBUG)
    finally:
        if DEBUG and _log_fp is not None:
            if _orig_stdout is not None:
                sys.stdout = _orig_stdout
            if _orig_stderr is not None:
                sys.stderr = _orig_stderr
            _log_fp.close()
