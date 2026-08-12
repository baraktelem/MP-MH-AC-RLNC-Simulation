"""
JamMpMhNetwork simulation driver.

Mirrors scripts/mp_mh_simulation.py but focuses on parameters specific to
JamMpMhNetwork (jammer_k, jammer_alpha). Five modes selectable at the
top of _run_main:

    MODE = "sweep_k"             - sweep jammer_k for fixed (epsilon, alpha)
                                   and plot throughput / mean delay / max
                                   delay vs k. The headline experiment.

    MODE = "sweep_k_multi_eps"   - same as sweep_k but overlays one labeled
                                   line per (e1, e2) pair on each subplot.

    MODE = "sweep_eps_grid_per_k" - same (e1, e2) sweep convention as
                                   mp_mh_simulation.py (8x8 grid via
                                   np.arange(0.1, 0.9, 0.1)) for each k.
                                   Plots one 3D surface per k value
                                   overlaid on each of the three metric
                                   subplots. The k=0 surface is directly
                                   comparable to mp_mh_simulation.py's
                                   output (same axes, same convention).

    MODE = "sweep_alpha"         - sweep jammer_alpha for fixed (epsilon, k)
                                   and plot the same three metrics vs alpha.

    MODE = "validate_k0"         - at jammer_k = 0, run JamMpMhNetwork and
                                   MpMhNetwork on the same epsilon matrix
                                   and print mean +- std side-by-side.
                                   Cross-topology sanity check at matched
                                   eps (chains vs layered cascade).

Recommended parameter sweeps:
  - sweep_k: K_VALUES = list(range(0, NUM_PATHS * NUM_HOPS + 1)). Expect
    monotonic degradation; throughput should hit 0 at k = P*H.
  - sweep_k_multi_eps: 3-4 (e1, e2) pairs spanning low/mid/high loss.
  - sweep_eps_grid_per_k: EPS_GRID_VALUES = np.arange(0.1, 0.9, 0.1) and a
    handful of k values (e.g., [0, 2, 4, 6, 8, 10, 12]). NUM_ITERATIONS=10-20
    keeps total runtime tractable (sims = 64 * len(K) * NUM_ITERATIONS).
  - sweep_alpha: ALPHA_VALUES = [1, 2, 3, 4, 6, 12]. Higher alpha means
    shorter jamming rounds (RTT/alpha).
  - validate_k0: NUM_ITERATIONS ~ 50, paper template at e1=0.1 e2=0.2.

Run:
    python scripts/jam_mp_mh_simulation.py

Set LOAD_EXISTING = True to replot from <RESULTS_FILE> without re-running.
"""

from __future__ import annotations

import os
import pickle
import statistics
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "jamming_simulation"))

from mh_epsilon_matrix import (
    DEFAULT_NUM_HOPS,
    DEFAULT_NUM_PATHS,
    article_matrix_for_hops,
    build_path_epsilons,
    print_article_epsilon_matrix,
    validate_article_matrix,
)
from Network import MpMhNetwork, SimulationStats
from JamNetwork import JamMpMhNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chain_major_epsilons(e1: float, e2: float, num_hops: int) -> list[list[float]]:
    """Paper-style ε matrix in chain-major form path_epsilons[c][h]."""
    article = article_matrix_for_hops(e1, e2, num_hops)  # already [path][hop]
    validate_article_matrix(article, DEFAULT_NUM_PATHS, num_hops)
    return [list(row) for row in article]


def uniform_chain_major_epsilons(eps: float, num_paths: int, num_hops: int) -> list[list[float]]:
    return [[eps for _ in range(num_hops)] for _ in range(num_paths)]


def chain_min_cut_capacity(e1: float, e2: float, num_hops_eff: int = 3) -> float:
    """Per-chain bottleneck bound: sum_c min_h(1 - eps[c][h]) for the paper
    article template at (e1, e2). NOT used as the default capacity reference,
    because it ignores the cross-chain coding gain that AC-RLNC achieves at
    the receiver -- empirically `k=0` throughput exceeds this number, so it
    is a *too-pessimistic* bound for our protocol. Kept here for reference."""
    article = article_matrix_for_hops(e1, e2, num_hops_eff)
    return sum(min(1.0 - eps for eps in chain) for chain in article)


# The "real" capacity reference: the same MpMhNetwork min-cut used by
# mp_mh_simulation.py. Computed via a layered BEC max-flow and accounts for
# cross-path coding gain. Wrapped here so callers don't need to know the
# exact module path; gracefully degrades to None if NetworkX is unavailable.
try:
    from mh_min_cut_capacity import min_cut_capacity_for_epsilons as _mp_mh_min_cut
except Exception:  # pragma: no cover -- NetworkX missing
    _mp_mh_min_cut = None


def network_min_cut_capacity(e1: float, e2: float, num_hops_eff: int = 3) -> float | None:
    """The MpMhNetwork min-cut capacity for the paper article template at
    (e1, e2). Same function used by mp_mh_simulation.py for the red reference
    surface. Returns None if NetworkX (its dependency) isn't installed."""
    if _mp_mh_min_cut is None:
        return None
    return float(_mp_mh_min_cut(e1, e2, num_hops_eff))


def _zero_stats(num_packets_sent: int = 0) -> SimulationStats:
    """Synthesize zeroed stats for the all-jammed / no-decoded case so we
    do not trip the pre-existing ZeroDivisionError in
    Network.calculate_inorder_delays_stats when 0 packets decode."""
    return SimulationStats(
        normalized_throughput=0.0,
        inorder_delay_mean=0.0,
        inorder_delay_max=0,
        num_new_rlnc_packets=0,
        num_fec_packets=0,
        num_fb_fec_packets=0,
        num_transmissions=0,
        num_information_packets_sent=num_packets_sent,
        num_information_packets_decoded=0,
        num_transmissions_dropped=0,
    )


def run_jam_network(
    *,
    path_eps_chain_major: list[list[float]],
    num_paths: int,
    num_hops: int,
    rtt: int,
    threshold: float,
    o_bar: int,
    num_packets_to_send: int,
    max_iterations: int | None,
    jammer_alpha: int,
    jammer_k: int,
    initial_epsilon: float = 0.5,
    debug: bool = False,
) -> SimulationStats:
    """Run one JamMpMhNetwork simulation. Returns zeroed SimulationStats if
    no packets decoded (avoids the pre-existing ZeroDivisionError)."""
    network = JamMpMhNetwork(
        path_epsilons=path_eps_chain_major,
        initial_epsilon=initial_epsilon,
        max_iterations=max_iterations,
        num_packets_to_send=num_packets_to_send,
        max_allowed_overlap=o_bar,
        num_paths=num_paths,
        global_prop_delay=rtt // 2,
        threshold=threshold,
        num_hops=num_hops,
        jammer_alpha=jammer_alpha,
        jammer_k=jammer_k,
        debug=debug,
    )

    if max_iterations is None:
        # No bounded mode without max_iterations -- caller should pass one
        # for sweeps so an all-jammed run terminates.
        network.run_sim()
        return network.get_simulation_stats()

    # Run via _tick directly so we can detect the no-decoded case before
    # invoking collect_stats (which crashes when nothing decoded).
    for t in range(1, max_iterations + 1):
        network.t = t
        network._tick()
        if (
            len(network.receiver.information_packets_decoding_times)
            >= num_packets_to_send
        ):
            break

    decoded_count = len(network.receiver.information_packets_decoding_times)
    if decoded_count == 0:
        return _zero_stats(num_packets_sent=num_packets_to_send)
    network.collect_stats()
    return network.get_simulation_stats()


def run_mp_mh_network(
    *,
    e1: float,
    e2: float,
    num_paths: int,
    num_hops: int,
    rtt: int,
    threshold: float,
    o_bar: int,
    num_packets_to_send: int,
    max_iterations: int | None,
    initial_epsilon: float = 0.5,
    debug: bool = False,
) -> SimulationStats:
    """Run one MpMhNetwork simulation at (e1, e2) for direct comparison."""
    network = MpMhNetwork(
        path_epsilons=build_path_epsilons(e1, e2, num_hops),
        initial_epsilon=initial_epsilon,
        max_iterations=max_iterations,
        num_packets_to_send=num_packets_to_send,
        max_allowed_overlap=o_bar,
        num_paths=num_paths,
        global_prop_delay=rtt // 2,
        threshold=threshold,
        num_hops=num_hops,
        debug=debug,
    )
    network.run_sim()
    return network.get_simulation_stats()


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------

def aggregate_by_key(
    results: list[tuple[int | float, SimulationStats]],
) -> dict[int | float, dict]:
    grouped: defaultdict = defaultdict(
        lambda: {"throughput": [], "delay_mean": [], "delay_max": []}
    )
    for key, stats in results:
        grouped[key]["throughput"].append(stats.normalized_throughput)
        grouped[key]["delay_mean"].append(stats.inorder_delay_mean)
        grouped[key]["delay_max"].append(stats.inorder_delay_max)

    aggregated: dict[int | float, dict] = {}
    for key, vals in grouped.items():
        aggregated[key] = {
            "throughput_mean": float(np.mean(vals["throughput"])),
            "throughput_std": float(np.std(vals["throughput"])),
            "delay_mean_mean": float(np.mean(vals["delay_mean"])),
            "delay_mean_std": float(np.std(vals["delay_mean"])),
            "delay_max_mean": float(np.mean(vals["delay_max"])),
            "delay_max_std": float(np.std(vals["delay_max"])),
            "n": len(vals["throughput"]),
        }
    return aggregated


def plot_sweep(
    aggregated: dict[int | float, dict],
    *,
    x_label: str,
    title_suffix: str,
    plot_path: str,
) -> None:
    keys = sorted(aggregated.keys())
    tp_mean = [aggregated[k]["throughput_mean"] for k in keys]
    tp_std = [aggregated[k]["throughput_std"] for k in keys]
    dm_mean = [aggregated[k]["delay_mean_mean"] for k in keys]
    dm_std = [aggregated[k]["delay_mean_std"] for k in keys]
    dx_mean = [aggregated[k]["delay_max_mean"] for k in keys]
    dx_std = [aggregated[k]["delay_max_std"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].errorbar(keys, tp_mean, yerr=tp_std, marker="o", capsize=3, color="tab:blue")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Normalized Throughput")
    axes[0].set_title("Throughput vs " + x_label)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].errorbar(keys, dm_mean, yerr=dm_std, marker="o", capsize=3, color="tab:orange")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Mean In-Order Delay")
    axes[1].set_title("Mean Delay vs " + x_label)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    axes[2].errorbar(keys, dx_mean, yerr=dx_std, marker="o", capsize=3, color="tab:red")
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Max In-Order Delay")
    axes[2].set_title("Max Delay vs " + x_label)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(bottom=0)

    fig.suptitle(title_suffix, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Plot saved to: {plot_path}")
    plt.show()


def plot_sweep_multi(
    series: dict[str, dict[int | float, dict]],
    *,
    x_label: str,
    title_suffix: str,
    plot_path: str,
) -> None:
    """Overlay multiple sweep curves on the same 3 subplots.

    `series` maps series-label -> aggregated dict (output of aggregate_by_key).
    Each label becomes one line per subplot, with shared x-axis values across
    all series (unique union of all keys).
    """
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for idx, (label, aggregated) in enumerate(series.items()):
        color = cmap(idx % 10)
        keys = sorted(aggregated.keys())
        tp_mean = [aggregated[k]["throughput_mean"] for k in keys]
        tp_std = [aggregated[k]["throughput_std"] for k in keys]
        dm_mean = [aggregated[k]["delay_mean_mean"] for k in keys]
        dm_std = [aggregated[k]["delay_mean_std"] for k in keys]
        dx_mean = [aggregated[k]["delay_max_mean"] for k in keys]
        dx_std = [aggregated[k]["delay_max_std"] for k in keys]

        axes[0].errorbar(keys, tp_mean, yerr=tp_std, marker="o", capsize=3,
                         color=color, label=label)
        axes[1].errorbar(keys, dm_mean, yerr=dm_std, marker="o", capsize=3,
                         color=color, label=label)
        axes[2].errorbar(keys, dx_mean, yerr=dx_std, marker="o", capsize=3,
                         color=color, label=label)

    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Normalized Throughput")
    axes[0].set_title("Throughput vs " + x_label)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Mean In-Order Delay")
    axes[1].set_title("Mean Delay vs " + x_label)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=9)

    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Max In-Order Delay")
    axes[2].set_title("Max Delay vs " + x_label)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(bottom=0)
    axes[2].legend(fontsize=9)

    fig.suptitle(title_suffix, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Plot saved to: {plot_path}")
    plt.show()


def plot_per_k_surfaces(
    aggregated_per_k: dict[int, dict[tuple[float, float], dict]],
    *,
    eps_values_e1: list[float],
    eps_values_e2: list[float],
    title_suffix: str,
    plot_path: str,
    capacity_func=None,
    capacity_label: str = "Chain min-cut capacity",
    capacity_color: str = "red",
) -> None:
    """Three 3D subplots (throughput / mean delay / max delay). Each subplot
    overlays one alpha-blended surface per k value (distinct color per k)
    on the same (eps1, eps2) base grid. Mirrors mp_mh_simulation.py's
    surface style so JamMpMhNetwork(k=0) can be visually compared against
    MpMhNetwork output produced by mp_mh_simulation.py.

    capacity_func (optional): callable (e1, e2) -> float that returns a
    capacity bound. Plotted as a transparent reference surface ON THE
    THROUGHPUT SUBPLOT ONLY. For JamMpMhNetwork, pass `chain_min_cut_capacity`.
    """
    n_e1 = len(eps_values_e1)
    n_e2 = len(eps_values_e2)
    EPS1, EPS2 = np.meshgrid(eps_values_e1, eps_values_e2)

    sorted_ks = sorted(aggregated_per_k.keys())
    cmap = plt.get_cmap("viridis")
    color_for = (
        lambda i: cmap(i / max(1, len(sorted_ks) - 1))
        if len(sorted_ks) > 1
        else cmap(0.5)
    )

    metrics = [
        ("throughput_mean", "Normalized Throughput"),
        ("delay_mean_mean", "Mean In-Order Delay"),
        ("delay_max_mean", "Max In-Order Delay"),
    ]

    fig = plt.figure(figsize=(20, 6))
    for subplot_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = fig.add_subplot(1, 3, subplot_idx + 1, projection="3d")
        max_z = 0.0
        for k_idx, k in enumerate(sorted_ks):
            agg = aggregated_per_k[k]
            grid = np.zeros((n_e1, n_e2))
            for i, e1 in enumerate(eps_values_e1):
                for j, e2 in enumerate(eps_values_e2):
                    grid[i, j] = agg.get((e1, e2), {}).get(metric_key, 0.0)
            max_z = max(max_z, float(np.max(grid)))
            ax.plot_surface(
                EPS1,
                EPS2,
                grid.T,
                color=color_for(k_idx),
                edgecolor="none",
                alpha=0.4,
            )

        legend_patches = [
            Patch(color=color_for(i), label=f"k={k}", alpha=0.4)
            for i, k in enumerate(sorted_ks)
        ]

        # Capacity reference surface goes only on the throughput subplot
        if subplot_idx == 0 and capacity_func is not None:
            cap_grid = np.zeros((n_e1, n_e2))
            for i, e1 in enumerate(eps_values_e1):
                for j, e2 in enumerate(eps_values_e2):
                    cap_grid[i, j] = float(capacity_func(e1, e2))
            max_z = max(max_z, float(np.max(cap_grid)))
            ax.plot_surface(
                EPS1,
                EPS2,
                cap_grid.T,
                color=capacity_color,
                edgecolor="none",
                alpha=0.2,
            )
            legend_patches.append(
                Patch(color=capacity_color, label=capacity_label, alpha=0.2)
            )

        ax.set_xlabel("ε₁")
        ax.set_ylabel("ε₂")
        ax.set_zlabel(metric_label)
        ax.set_title(f"{metric_label} (1 surface per k)")
        ax.set_zlim(0, max(0.1, max_z * 1.1))
        ax.view_init(elev=20, azim=45)
        ax.legend(handles=legend_patches, fontsize=8, loc="upper left")

    fig.suptitle(title_suffix, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Plot saved to: {plot_path}")
    plt.show()


def save_pickle(obj, filename: str | None, prefix: str) -> str:
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"[OK] Results saved to: {filename}")
    return filename


def load_pickle(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file not found: {filename}")
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    print(f"[OK] Results loaded from: {filename}")
    return obj


# ---------------------------------------------------------------------------
# Parallel sim runner
# ---------------------------------------------------------------------------

def _run_one_sim_task(args: tuple) -> tuple:
    """Top-level worker for ProcessPoolExecutor (must be picklable -- defined at
    module level). Runs one JamMpMhNetwork sim and returns its result tagged
    with the original task identity.

    args: (group_key, k, e1, e2, it, alpha,
           num_paths, num_hops, rtt, threshold, o_bar,
           num_packets_to_send, max_iterations, initial_epsilon)
    """
    (
        group_key,
        k,
        e1,
        e2,
        it,
        alpha,
        num_paths,
        num_hops,
        rtt,
        threshold,
        o_bar,
        num_packets_to_send,
        max_iterations,
        initial_epsilon,
    ) = args
    eps_matrix = chain_major_epsilons(e1, e2, num_hops)
    stats = run_jam_network(
        path_eps_chain_major=eps_matrix,
        num_paths=num_paths,
        num_hops=num_hops,
        rtt=rtt,
        threshold=threshold,
        o_bar=o_bar,
        num_packets_to_send=num_packets_to_send,
        max_iterations=max_iterations,
        jammer_alpha=alpha,
        jammer_k=k,
        initial_epsilon=initial_epsilon,
        debug=False,
    )
    return (group_key, k, float(e1), float(e2), it, alpha, stats)


def _execute_tasks(
    tasks: list[tuple],
    *,
    parallel_workers: int,
    progress_label: str,
):
    """Execute sim tasks either serially (parallel_workers <= 1) or via a
    ProcessPoolExecutor. Yields (task_idx, result_tuple) as results complete.
    With parallel execution, results arrive in arbitrary order; the caller
    must use group_key inside result_tuple to route results to the right
    bucket. Prints a per-task progress line tagged with the order index."""
    total = len(tasks)
    print(
        f"\n[{progress_label}] running {total} sims with parallel_workers={parallel_workers}\n"
    )

    if parallel_workers is None or parallel_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            yield idx, _run_one_sim_task(task)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as pool:
            future_to_idx = {
                pool.submit(_run_one_sim_task, task): idx
                for idx, task in enumerate(tasks, start=1)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                yield idx, future.result()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_sweep_k(
    *,
    e1: float,
    e2: float,
    num_paths: int,
    num_hops: int,
    rtt: int,
    threshold: float,
    o_bar: int,
    num_packets_to_send: int,
    max_iterations: int,
    jammer_alpha: int,
    k_values: list[int],
    num_iterations: int,
    initial_epsilon: float,
    results_file: str,
    plot_file: str,
    load_existing: bool,
    parallel_workers: int = 1,
) -> None:
    if load_existing and os.path.exists(results_file):
        results = load_pickle(results_file)
    else:
        tasks: list[tuple] = []
        for it in range(1, num_iterations + 1):
            for k in k_values:
                tasks.append((
                    k,
                    k,
                    float(e1),
                    float(e2),
                    it,
                    jammer_alpha,
                    num_paths,
                    num_hops,
                    rtt,
                    threshold,
                    o_bar,
                    num_packets_to_send,
                    max_iterations,
                    initial_epsilon,
                ))

        results: list[tuple[int, SimulationStats]] = []
        total = len(tasks)
        completed = 0
        for _, result in _execute_tasks(
            tasks,
            parallel_workers=parallel_workers,
            progress_label="sweep_k",
        ):
            (_gk, k, _e1, _e2, it, _alpha, stats) = result
            results.append((k, stats))
            completed += 1
            print(
                f"[{completed}/{total}] iter {it}/{num_iterations} k={k:>2} "
                f"-> tp={stats.normalized_throughput:.4f} "
                f"delay_mean={stats.inorder_delay_mean:.2f} "
                f"decoded={stats.num_information_packets_decoded}"
            )
        save_pickle(results, results_file, prefix="jam_sweep_k")

    aggregated = aggregate_by_key(results)
    plot_sweep(
        aggregated,
        x_label="Jammer k (paths blocked per round)",
        title_suffix=(
            f"JamMpMhNetwork sweep_k (H={num_hops}, P={num_paths}, "
            f"alpha={jammer_alpha}, RTT={rtt}, eps_template e1={e1} e2={e2})"
        ),
        plot_path=plot_file,
    )


def mode_sweep_alpha(
    *,
    e1: float,
    e2: float,
    num_paths: int,
    num_hops: int,
    rtt: int,
    threshold: float,
    o_bar: int,
    num_packets_to_send: int,
    max_iterations: int,
    jammer_k: int,
    alpha_values: list[int],
    num_iterations: int,
    initial_epsilon: float,
    results_file: str,
    plot_file: str,
    load_existing: bool,
) -> None:
    if load_existing and os.path.exists(results_file):
        results = load_pickle(results_file)
    else:
        eps = chain_major_epsilons(e1, e2, num_hops)
        results: list[tuple[int, SimulationStats]] = []
        total = num_iterations * len(alpha_values)
        sim = 0
        for it in range(1, num_iterations + 1):
            for alpha in alpha_values:
                sim += 1
                stats = run_jam_network(
                    path_eps_chain_major=eps,
                    num_paths=num_paths,
                    num_hops=num_hops,
                    rtt=rtt,
                    threshold=threshold,
                    o_bar=o_bar,
                    num_packets_to_send=num_packets_to_send,
                    max_iterations=max_iterations,
                    jammer_alpha=alpha,
                    jammer_k=jammer_k,
                    initial_epsilon=initial_epsilon,
                    debug=False,
                )
                results.append((alpha, stats))
                print(
                    f"[{sim}/{total}] iter {it}/{num_iterations} alpha={alpha:>2} "
                    f"-> tp={stats.normalized_throughput:.4f} "
                    f"delay_mean={stats.inorder_delay_mean:.2f}"
                )
        save_pickle(results, results_file, prefix="jam_sweep_alpha")

    aggregated = aggregate_by_key(results)
    plot_sweep(
        aggregated,
        x_label="Jammer alpha (jamming round = RTT/alpha)",
        title_suffix=(
            f"JamMpMhNetwork sweep_alpha (H={num_hops}, P={num_paths}, "
            f"k={jammer_k}, RTT={rtt}, eps_template e1={e1} e2={e2})"
        ),
        plot_path=plot_file,
    )


def mode_validate_k0(
    *,
    e1: float,
    e2: float,
    num_paths: int,
    num_hops: int,
    rtt: int,
    threshold: float,
    o_bar: int,
    num_packets_to_send: int,
    max_iterations: int | None,
    num_iterations: int,
    initial_epsilon: float,
    results_file: str,
    load_existing: bool,
) -> None:
    """Run JamMpMhNetwork with jammer_k=0 and MpMhNetwork at the same ε.
    Both use the paper template, with the same loss matrix per chain/path.
    With k=0 the Jammer never blocks anything, so any difference reflects
    the topology change (independent chains vs layered cascade)."""
    if load_existing and os.path.exists(results_file):
        results = load_pickle(results_file)
    else:
        chain_eps = chain_major_epsilons(e1, e2, num_hops)
        results = {"jam_k0": [], "mpmh": []}
        total = num_iterations * 2
        sim = 0
        for it in range(1, num_iterations + 1):
            sim += 1
            jam_stats = run_jam_network(
                path_eps_chain_major=chain_eps,
                num_paths=num_paths,
                num_hops=num_hops,
                rtt=rtt,
                threshold=threshold,
                o_bar=o_bar,
                num_packets_to_send=num_packets_to_send,
                max_iterations=max_iterations,
                jammer_alpha=2,
                jammer_k=0,
                initial_epsilon=initial_epsilon,
                debug=False,
            )
            results["jam_k0"].append(jam_stats)
            print(
                f"[{sim}/{total}] iter {it}/{num_iterations} JamMpMhNetwork(k=0) "
                f"-> tp={jam_stats.normalized_throughput:.4f} "
                f"delay_mean={jam_stats.inorder_delay_mean:.2f}"
            )
            sim += 1
            mp_stats = run_mp_mh_network(
                e1=e1,
                e2=e2,
                num_paths=num_paths,
                num_hops=num_hops,
                rtt=rtt,
                threshold=threshold,
                o_bar=o_bar,
                num_packets_to_send=num_packets_to_send,
                max_iterations=max_iterations,
                initial_epsilon=initial_epsilon,
                debug=False,
            )
            results["mpmh"].append(mp_stats)
            print(
                f"[{sim}/{total}] iter {it}/{num_iterations} MpMhNetwork           "
                f"-> tp={mp_stats.normalized_throughput:.4f} "
                f"delay_mean={mp_stats.inorder_delay_mean:.2f}"
            )
        save_pickle(results, results_file, prefix="jam_validate_k0")

    def _summarize(label: str, stats_list: list[SimulationStats]) -> None:
        tps = [s.normalized_throughput for s in stats_list]
        dms = [s.inorder_delay_mean for s in stats_list]
        dxs = [s.inorder_delay_max for s in stats_list]
        print(
            f"  {label:<22} "
            f"throughput {statistics.mean(tps):.4f} +- {statistics.pstdev(tps):.4f}  |  "
            f"delay_mean {statistics.mean(dms):.2f} +- {statistics.pstdev(dms):.2f}  |  "
            f"delay_max {statistics.mean(dxs):.2f} +- {statistics.pstdev(dxs):.2f}  "
            f"(n={len(stats_list)})"
        )

    print("\n" + "=" * 70)
    print("VALIDATE k=0: JamMpMhNetwork(k=0) vs MpMhNetwork at matched eps")
    print("=" * 70)
    _summarize("JamMpMhNetwork(k=0)", results["jam_k0"])
    _summarize("MpMhNetwork", results["mpmh"])
    print(
        "\nNote: with k=0 the Jammer never blocks. Differences reflect the "
        "topology change (P independent single-path chains vs layered cascade "
        "with per-hop mixing). Throughput should be in the same ballpark; "
        "JamMpMhNetwork is typically a touch lower because it cannot mix "
        "across chains at intermediate Nodes."
    )


def mode_sweep_k_multi_eps(
    *,
    eps_pairs: list[tuple[float, float]],
    num_paths: int,
    num_hops: int,
    rtt: int,
    threshold: float,
    o_bar: int,
    num_packets_to_send: int,
    max_iterations: int,
    jammer_alpha: int,
    k_values: list[int],
    num_iterations: int,
    initial_epsilon: float,
    results_file: str,
    plot_file: str,
    load_existing: bool,
    parallel_workers: int = 1,
) -> None:
    """Sweep jammer_k for each (e1, e2) in eps_pairs and overlay all curves
    on one figure (3 subplots: throughput / mean delay / max delay)."""
    if load_existing and os.path.exists(results_file):
        per_eps_results = load_pickle(results_file)
    else:
        per_eps_target_count = num_iterations * len(k_values)
        tasks: list[tuple] = []
        for (e1, e2) in eps_pairs:
            for it in range(1, num_iterations + 1):
                for k in k_values:
                    tasks.append((
                        (float(e1), float(e2)),  # group_key
                        k,
                        float(e1),
                        float(e2),
                        it,
                        jammer_alpha,
                        num_paths,
                        num_hops,
                        rtt,
                        threshold,
                        o_bar,
                        num_packets_to_send,
                        max_iterations,
                        initial_epsilon,
                    ))

        per_eps_results: dict[tuple[float, float], list[tuple[int, SimulationStats]]] = {
            (float(e1), float(e2)): [] for (e1, e2) in eps_pairs
        }
        per_eps_complete: dict[tuple[float, float], int] = {
            (float(e1), float(e2)): 0 for (e1, e2) in eps_pairs
        }
        total = len(tasks)
        completed = 0
        for _, result in _execute_tasks(
            tasks,
            parallel_workers=parallel_workers,
            progress_label="sweep_k_multi_eps",
        ):
            (group_key, k, e1, e2, it, _alpha, stats) = result
            per_eps_results[group_key].append((k, stats))
            per_eps_complete[group_key] += 1
            completed += 1
            print(
                f"[{completed}/{total}] e1={e1:.2f} e2={e2:.2f} "
                f"iter {it}/{num_iterations} k={k:>2} "
                f"-> tp={stats.normalized_throughput:.4f} "
                f"delay_mean={stats.inorder_delay_mean:.2f}"
            )
            if per_eps_complete[group_key] == per_eps_target_count:
                save_pickle(per_eps_results, results_file, prefix="jam_sweep_k_multi_eps")
                print(f"[checkpoint] (e1={e1:.2f}, e2={e2:.2f}) complete -> saved partial pickle")
        save_pickle(per_eps_results, results_file, prefix="jam_sweep_k_multi_eps")

    series: dict[str, dict[int | float, dict]] = {}
    for (e1, e2), results in per_eps_results.items():
        series[f"e1={e1:.2f}, e2={e2:.2f}"] = aggregate_by_key(results)

    plot_sweep_multi(
        series,
        x_label="Jammer k (paths blocked per round)",
        title_suffix=(
            f"JamMpMhNetwork sweep_k_multi_eps "
            f"(H={num_hops}, P={num_paths}, alpha={jammer_alpha}, RTT={rtt})"
        ),
        plot_path=plot_file,
    )


def mode_sweep_eps_grid_per_k(
    *,
    eps_values: list[float],
    k_values: list[int],
    num_paths: int,
    num_hops: int,
    rtt: int,
    threshold: float,
    o_bar: int,
    num_packets_to_send: int,
    max_iterations: int,
    jammer_alpha: int,
    num_iterations: int,
    initial_epsilon: float,
    results_file: str,
    plot_file: str,
    load_existing: bool,
    parallel_workers: int = 1,
) -> None:
    """Sweep (e1, e2) over the eps_values × eps_values grid (mp_mh_simulation.py
    convention) for each k in k_values, and plot one 3D surface per k overlaid
    on the same (eps1, eps2) plane for throughput / mean delay / max delay.

    With k=0 the resulting surface is directly comparable to the surfaces
    produced by mp_mh_simulation.py; differences reflect the chain topology
    (independent chains vs layered cascade).

    Set parallel_workers > 1 to run sims across CPU cores (uses
    concurrent.futures.ProcessPoolExecutor). Pickle is saved incrementally
    after each k completes, so a crash only loses the in-progress k."""
    eps_sorted = sorted(eps_values)

    if load_existing and os.path.exists(results_file):
        per_k_results = load_pickle(results_file)
    else:
        per_k_target_count = num_iterations * len(eps_sorted) ** 2
        tasks: list[tuple] = []
        for k in k_values:
            for it in range(1, num_iterations + 1):
                for e1 in eps_sorted:
                    for e2 in eps_sorted:
                        tasks.append((
                            k,                       # group_key
                            k,
                            float(e1),
                            float(e2),
                            it,
                            jammer_alpha,
                            num_paths,
                            num_hops,
                            rtt,
                            threshold,
                            o_bar,
                            num_packets_to_send,
                            max_iterations,
                            initial_epsilon,
                        ))

        per_k_results: dict[int, list[tuple[float, float, SimulationStats]]] = {
            k: [] for k in k_values
        }
        per_k_complete: dict[int, int] = {k: 0 for k in k_values}
        total = len(tasks)
        completed = 0

        for _, result in _execute_tasks(
            tasks,
            parallel_workers=parallel_workers,
            progress_label="sweep_eps_grid_per_k",
        ):
            (group_key, k, e1, e2, it, _alpha, stats) = result
            per_k_results[group_key].append((e1, e2, stats))
            per_k_complete[group_key] += 1
            completed += 1
            print(
                f"[{completed}/{total}] k={k:>2} iter {it}/{num_iterations} "
                f"e1={e1:.1f} e2={e2:.1f} -> "
                f"tp={stats.normalized_throughput:.4f} "
                f"delay_mean={stats.inorder_delay_mean:.2f}"
            )
            if per_k_complete[group_key] == per_k_target_count:
                save_pickle(per_k_results, results_file, prefix="jam_sweep_eps_grid_per_k")
                print(f"[checkpoint] k={group_key} complete -> saved partial pickle")

        save_pickle(per_k_results, results_file, prefix="jam_sweep_eps_grid_per_k")

    aggregated_per_k: dict[int, dict[tuple[float, float], dict]] = {}
    for k, results in per_k_results.items():
        grouped: defaultdict = defaultdict(
            lambda: {"throughput": [], "delay_mean": [], "delay_max": []}
        )
        for e1, e2, stats in results:
            grouped[(float(e1), float(e2))]["throughput"].append(stats.normalized_throughput)
            grouped[(float(e1), float(e2))]["delay_mean"].append(stats.inorder_delay_mean)
            grouped[(float(e1), float(e2))]["delay_max"].append(stats.inorder_delay_max)
        aggregated_per_k[k] = {
            key: {
                "throughput_mean": float(np.mean(v["throughput"])),
                "throughput_std": float(np.std(v["throughput"])),
                "delay_mean_mean": float(np.mean(v["delay_mean"])),
                "delay_mean_std": float(np.std(v["delay_mean"])),
                "delay_max_mean": float(np.mean(v["delay_max"])),
                "delay_max_std": float(np.std(v["delay_max"])),
                "n": len(v["throughput"]),
            }
            for key, v in grouped.items()
        }

    # Use the same NetworkX min-cut as mp_mh_simulation.py so the capacity
    # reference is directly comparable. Falls back to no capacity surface if
    # NetworkX isn't installed.
    capacity_func = None
    capacity_label = "Min-cut capacity"
    if _mp_mh_min_cut is not None:
        capacity_func = lambda e1, e2: float(_mp_mh_min_cut(e1, e2, num_hops))
        capacity_label = "Min-cut capacity (NetworkX, layered BEC)"

    plot_per_k_surfaces(
        aggregated_per_k,
        eps_values_e1=eps_sorted,
        eps_values_e2=eps_sorted,
        title_suffix=(
            f"JamMpMhNetwork sweep_eps_grid_per_k (H={num_hops}, P={num_paths}, "
            f"alpha={jammer_alpha}, RTT={rtt}); compare k=0 surface to mp_mh_simulation.py output"
        ),
        plot_path=plot_file,
        capacity_func=capacity_func,
        capacity_label=capacity_label,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_main() -> None:
    print("=" * 70)
    print(" " * 8 + "MP-MH AC-RLNC -- JamMpMhNetwork simulation driver")
    print("=" * 70)

    # ---- Topology / protocol constants (mirrors mp_mh_simulation.py) ------
    NUM_PATHS = DEFAULT_NUM_PATHS                          # 4
    NUM_HOPS = DEFAULT_NUM_HOPS                            # 3
    RTT = 12
    PROP_DELAY = RTT // 2
    THRESHOLD = 0.0
    O_BAR = 2 * NUM_PATHS * (RTT - 1)
    NUM_PACKETS_TO_SEND = 500
    MAX_ITERATIONS = 4000
    NUM_ITERATIONS = 150
    INITIAL_EPSILON = 0.5
    EPS_E1 = 0.1
    EPS_E2 = 0.2

    # ---- Mode selection --------------------------------------------------
    MODE = "sweep_eps_grid_per_k"  # one of: "sweep_k", "sweep_k_multi_eps", "sweep_eps_grid_per_k", "sweep_alpha", "validate_k0"
    LOAD_EXISTING = False

    # Parallelization for the heavy sweep modes (sweep_k, sweep_k_multi_eps,
    # sweep_eps_grid_per_k). 1 = serial. A safe default is os.cpu_count() // 2
    # to leave headroom for the OS / other apps. Set to os.cpu_count() to use
    # every core. Has no effect on sweep_alpha / validate_k0.
    PARALLEL_WORKERS = max(1, (os.cpu_count() or 2) // 2)

    # sweep_k config
    K_VALUES: list[int] = list(range(0, NUM_PATHS * NUM_HOPS + 1))
    SWEEP_K_ALPHA = 2
    K_RESULTS_FILE = "jam_sweep_k_results.pkl"
    K_PLOT_FILE = "jam_sweep_k.png"

    # sweep_k_multi_eps config: each (e1, e2) becomes one labeled line in the plots
    MULTI_EPS_PAIRS: list[tuple[float, float]] = [
        (0.05, 0.05),  # low loss
        (0.1, 0.2),    # paper template
        (0.3, 0.3),    # mid loss
        (0.5, 0.5),    # high loss
    ]
    MULTI_EPS_RESULTS_FILE = "jam_sweep_k_multi_eps_results.pkl"
    MULTI_EPS_PLOT_FILE = "jam_sweep_k_multi_eps.png"

    # sweep_eps_grid_per_k config: same (e1, e2) sweep convention as
    # mp_mh_simulation.py (8x8 grid via np.arange(0.1, 0.9, 0.1)),
    # one 3D surface per k overlaid on each of the three metric subplots.
    EPS_GRID_VALUES: list[float] = [round(float(v), 2) for v in np.arange(0.1, 0.9, 0.1)]
    # EPS_GRID_K_VALUES: list[int] = [0, 2, 4, 6, 8, 10, 12]
    tot_paths = NUM_HOPS * NUM_PATHS
    EPS_GRID_K_VALUES: list[int] = [0, int(0.25 * tot_paths), int(0.5 * tot_paths), int(0.75 * tot_paths)]
    EPS_GRID_RESULTS_FILE = "jam_sweep_eps_grid_per_k_results.pkl"
    EPS_GRID_PLOT_FILE = "jam_sweep_eps_grid_per_k.png"

    # sweep_alpha config
    ALPHA_VALUES: list[int] = [1, 2, 3, 4, 6, 12]
    SWEEP_ALPHA_K = 2
    ALPHA_RESULTS_FILE = "jam_sweep_alpha_results.pkl"
    ALPHA_PLOT_FILE = "jam_sweep_alpha.png"

    # validate_k0 config
    VALIDATE_RESULTS_FILE = "jam_validate_k0_results.pkl"

    print("\nSimulation parameters:")
    print(f"  - RTT (slots): {RTT}, prop_delay: {PROP_DELAY}")
    print(f"  - Threshold: {THRESHOLD}, o_bar: {O_BAR}")
    print(f"  - Paths P = {NUM_PATHS}, hops H = {NUM_HOPS}")
    print(f"  - Packets per run: {NUM_PACKETS_TO_SEND}, max_iterations: {MAX_ITERATIONS}")
    print(f"  - Outer iterations: {NUM_ITERATIONS}")
    print(f"  - eps template: e1={EPS_E1}, e2={EPS_E2}")
    print_article_epsilon_matrix(EPS_E1, EPS_E2, NUM_PATHS, NUM_HOPS)
    print(f"  - MODE: {MODE}\n")

    if MODE == "sweep_k":
        mode_sweep_k(
            e1=EPS_E1,
            e2=EPS_E2,
            num_paths=NUM_PATHS,
            num_hops=NUM_HOPS,
            rtt=RTT,
            threshold=THRESHOLD,
            o_bar=O_BAR,
            num_packets_to_send=NUM_PACKETS_TO_SEND,
            max_iterations=MAX_ITERATIONS,
            jammer_alpha=SWEEP_K_ALPHA,
            k_values=K_VALUES,
            num_iterations=NUM_ITERATIONS,
            initial_epsilon=INITIAL_EPSILON,
            results_file=K_RESULTS_FILE,
            plot_file=K_PLOT_FILE,
            load_existing=LOAD_EXISTING,
            parallel_workers=PARALLEL_WORKERS,
        )
    elif MODE == "sweep_k_multi_eps":
        mode_sweep_k_multi_eps(
            eps_pairs=MULTI_EPS_PAIRS,
            num_paths=NUM_PATHS,
            num_hops=NUM_HOPS,
            rtt=RTT,
            threshold=THRESHOLD,
            o_bar=O_BAR,
            num_packets_to_send=NUM_PACKETS_TO_SEND,
            max_iterations=MAX_ITERATIONS,
            jammer_alpha=SWEEP_K_ALPHA,
            k_values=K_VALUES,
            num_iterations=NUM_ITERATIONS,
            initial_epsilon=INITIAL_EPSILON,
            results_file=MULTI_EPS_RESULTS_FILE,
            plot_file=MULTI_EPS_PLOT_FILE,
            load_existing=LOAD_EXISTING,
            parallel_workers=PARALLEL_WORKERS,
        )
    elif MODE == "sweep_eps_grid_per_k":
        mode_sweep_eps_grid_per_k(
            eps_values=EPS_GRID_VALUES,
            k_values=EPS_GRID_K_VALUES,
            num_paths=NUM_PATHS,
            num_hops=NUM_HOPS,
            rtt=RTT,
            threshold=THRESHOLD,
            o_bar=O_BAR,
            num_packets_to_send=NUM_PACKETS_TO_SEND,
            max_iterations=MAX_ITERATIONS,
            jammer_alpha=SWEEP_K_ALPHA,
            num_iterations=NUM_ITERATIONS,
            initial_epsilon=INITIAL_EPSILON,
            results_file=EPS_GRID_RESULTS_FILE,
            plot_file=EPS_GRID_PLOT_FILE,
            load_existing=LOAD_EXISTING,
            parallel_workers=PARALLEL_WORKERS,
        )
    elif MODE == "sweep_alpha":
        mode_sweep_alpha(
            e1=EPS_E1,
            e2=EPS_E2,
            num_paths=NUM_PATHS,
            num_hops=NUM_HOPS,
            rtt=RTT,
            threshold=THRESHOLD,
            o_bar=O_BAR,
            num_packets_to_send=NUM_PACKETS_TO_SEND,
            max_iterations=MAX_ITERATIONS,
            jammer_k=SWEEP_ALPHA_K,
            alpha_values=ALPHA_VALUES,
            num_iterations=NUM_ITERATIONS,
            initial_epsilon=INITIAL_EPSILON,
            results_file=ALPHA_RESULTS_FILE,
            plot_file=ALPHA_PLOT_FILE,
            load_existing=LOAD_EXISTING,
        )
    elif MODE == "validate_k0":
        mode_validate_k0(
            e1=EPS_E1,
            e2=EPS_E2,
            num_paths=NUM_PATHS,
            num_hops=NUM_HOPS,
            rtt=RTT,
            threshold=THRESHOLD,
            o_bar=O_BAR,
            num_packets_to_send=NUM_PACKETS_TO_SEND,
            max_iterations=MAX_ITERATIONS,
            num_iterations=NUM_ITERATIONS,
            initial_epsilon=INITIAL_EPSILON,
            results_file=VALIDATE_RESULTS_FILE,
            load_existing=LOAD_EXISTING,
        )
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")

    print("\nDone.")


if __name__ == "__main__":
    _run_main()
