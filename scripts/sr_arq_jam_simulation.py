"""
SRJamMpMhNetwork simulation driver (uncoded SR-ARQ under jamming).

The SR-ARQ counterpart of scripts/jam_mp_mh_simulation.py. Same P-chains x H-hops
jammed topology and the same five experiment modes, but the protocol is uncoded
hop-by-hop / end-to-end Selective-Repeat ARQ (sr_arq/) instead of AC-RLNC. Exactly
ONE SR feedback mode runs per invocation (chosen by SR_FEEDBACK_MODE), mirroring
scripts/sr_arq_mpmh_simulation.py.

The epsilon convention matches jam_mp_mh_simulation.py (fixed chains = the paper's
raw article rows, chain_major_epsilons(e1, e2, H)), so SR-ARQ results are directly
comparable to the AC-RLNC jammed results at matched channels.

Modes (selected at the top of _run_main):

    MODE = "sweep_k"              - sweep jammer_k for fixed (epsilon, alpha) and
                                    plot throughput / mean delay / max delay vs k.
                                    The headline experiment.

    MODE = "sweep_k_multi_eps"    - same as sweep_k but overlays one labeled line
                                    per (e1, e2) pair on each subplot.

    MODE = "sweep_eps_grid_per_k" - (e1, e2) grid (np.arange(0.1, 0.9, 0.1)) for
                                    each k; one 3D surface per k on each of the
                                    three metric subplots.

    MODE = "sweep_alpha"          - sweep jammer_alpha for fixed (epsilon, k) and
                                    plot the same three metrics vs alpha.

    MODE = "validate_k0"          - at jammer_k = 0 run SRJamMpMhNetwork and plain
                                    SRMpMhNetwork on the same epsilon matrix and
                                    print mean +- std side-by-side (the jam wrapper
                                    with k=0 must reproduce the plain SR network).

SR-ARQ knobs (all threaded through every mode): SR_FEEDBACK_MODE, SR_WINDOW (auto:
end-to-end RTT for E2E modes, per-hop RTT for HBH), IN_ORDER_FORWARDING,
NODE_QUEUE_SIZE, PACKETS_PER_PATH.

Run:
    python scripts/sr_arq_jam_simulation.py

Set LOAD_EXISTING = True to replot from <RESULTS_FILE> without re-running.
"""

from __future__ import annotations

import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "jamming_simulation"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sr_arq"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from mp_mh_network.Network import SimulationStats
from sr_arq.SRNetwork import SRMpMhNetwork
from sr_arq.SRJamNetwork import SRJamMpMhNetwork
from sr_arq.sr_feedback import SRFeedbackMode

# Reuse the generic (protocol-agnostic) aggregation / plotting / pickle helpers and
# the paper-article epsilon builder straight from the AC-RLNC jam driver -- they
# operate on SimulationStats and (e1, e2)/k keys, so nothing about them is AC-RLNC
# specific. This keeps the two jam drivers' plots identical in style and directly
# comparable. (The all-jammed zeroed-stats case is handled inside
# SRJamMpMhNetwork.run_sim, so no _zero_stats helper is needed here.)
from jam_mp_mh_simulation import (
    _mp_mh_min_cut,
    aggregate_by_key,
    chain_major_epsilons,
    load_pickle,
    plot_per_k_surfaces,
    plot_sweep,
    plot_sweep_multi,
    save_pickle,
)


# ---------------------------------------------------------------------------
# Single-sim runners
# ---------------------------------------------------------------------------

def run_sr_jam_network(
    *,
    path_eps_chain_major: list[list[float]],
    num_paths: int,
    num_hops: int,
    rtt: int,
    num_packets_to_send: int,
    max_iterations: int | None,
    jammer_alpha: int,
    jammer_k: int,
    window: int | None,
    in_order_forwarding: bool,
    node_queue_size: int | None,
    packets_per_path: int | None,
    feedback_mode: SRFeedbackMode,
    debug: bool = False,
) -> SimulationStats:
    """Run one SRJamMpMhNetwork simulation. The network's run_sim handles the stop
    trigger (max_iterations / num_packets_to_send / packets_per_path) and publishes
    zeroed stats when nothing decodes (the all-jammed case), so this is a thin
    build-and-run wrapper (mirrors _run_srmpmh in sr_arq_mpmh_simulation.py)."""
    net = SRJamMpMhNetwork(
        path_epsilons=path_eps_chain_major,
        num_paths=num_paths,
        num_hops=num_hops,
        global_prop_delay=rtt // 2,   # rtt is the end-to-end RTT; split across hops
        num_packets_to_send=num_packets_to_send,
        max_iterations=max_iterations,
        window=window,
        in_order_forwarding=in_order_forwarding,
        node_queue_size=node_queue_size,
        packets_per_path=packets_per_path,
        feedback_mode=feedback_mode,
        jammer_alpha=jammer_alpha,
        jammer_k=jammer_k,
        debug=debug,
    )
    net.run_sim()
    return net.get_simulation_stats()


def run_sr_plain_network(
    *,
    path_eps_chain_major: list[list[float]],
    num_paths: int,
    num_hops: int,
    rtt: int,
    num_packets_to_send: int,
    max_iterations: int | None,
    window: int | None,
    in_order_forwarding: bool,
    node_queue_size: int | None,
    packets_per_path: int | None,
    feedback_mode: SRFeedbackMode,
    debug: bool = False,
) -> SimulationStats:
    """Run one plain (un-jammed) SRMpMhNetwork on the same epsilon matrix for the
    validate_k0 comparison. Uses SRMpMhNetwork.run_sim, which honors the same three
    stop triggers; with k=0 (the only validate case) every packet is delivered."""
    net = SRMpMhNetwork(
        path_epsilons=path_eps_chain_major,
        num_paths=num_paths,
        num_hops=num_hops,
        global_prop_delay=rtt // 2,
        num_packets_to_send=num_packets_to_send,
        max_iterations=max_iterations,
        window=window,
        in_order_forwarding=in_order_forwarding,
        node_queue_size=node_queue_size,
        packets_per_path=packets_per_path,
        feedback_mode=feedback_mode,
        debug=debug,
    )
    net.run_sim()
    return net.get_simulation_stats()


# ---------------------------------------------------------------------------
# Parallel sim runner (worker must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_one_sim_task(args: tuple) -> tuple:
    """Top-level worker for ProcessPoolExecutor. Runs one SRJamMpMhNetwork sim.

    args: (group_key, k, e1, e2, it, alpha,
           num_paths, num_hops, rtt, num_packets_to_send, max_iterations,
           window, in_order_forwarding, node_queue_size, packets_per_path,
           feedback_mode)
    returns: (group_key, k, float(e1), float(e2), it, alpha, stats)
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
        num_packets_to_send,
        max_iterations,
        window,
        in_order_forwarding,
        node_queue_size,
        packets_per_path,
        feedback_mode,
    ) = args
    eps_matrix = chain_major_epsilons(e1, e2, num_hops)
    stats = run_sr_jam_network(
        path_eps_chain_major=eps_matrix,
        num_paths=num_paths,
        num_hops=num_hops,
        rtt=rtt,
        num_packets_to_send=num_packets_to_send,
        max_iterations=max_iterations,
        jammer_alpha=alpha,
        jammer_k=k,
        window=window,
        in_order_forwarding=in_order_forwarding,
        node_queue_size=node_queue_size,
        packets_per_path=packets_per_path,
        feedback_mode=feedback_mode,
        debug=False,
    )
    return (group_key, k, float(e1), float(e2), it, alpha, stats)


def _execute_tasks(tasks: list[tuple], *, parallel_workers: int, progress_label: str):
    """Execute sim tasks serially (parallel_workers <= 1) or via a
    ProcessPoolExecutor. Yields (task_idx, result_tuple) as results complete.
    With parallel execution, results arrive in arbitrary order; the caller must
    use group_key inside result_tuple to route results to the right bucket.

    The pool is managed manually (poll with a 0.5s timeout) so Ctrl+C stops
    promptly on Windows, mirroring scripts/sr_arq_mpmh_simulation.py."""
    total = len(tasks)
    print(
        f"\n[{progress_label}] running {total} sims with parallel_workers={parallel_workers}\n"
    )

    if parallel_workers is None or parallel_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            yield idx, _run_one_sim_task(task)
        return

    pool = ProcessPoolExecutor(max_workers=parallel_workers)
    futures: list = []
    try:
        futures = [pool.submit(_run_one_sim_task, t) for t in tasks]
        future_to_idx = {f: i for i, f in enumerate(futures, start=1)}
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for fut in finished:
                yield future_to_idx[fut], fut.result()
    except (KeyboardInterrupt, GeneratorExit):
        print("\n[INTERRUPTED] Cancelling pending simulations and terminating workers...")
        for f in futures:
            f.cancel()
        for proc in list(getattr(pool, "_processes", {}).values()):
            proc.terminate()
        raise
    finally:
        pool.shutdown(wait=False)


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
    num_packets_to_send: int,
    max_iterations: int,
    jammer_alpha: int,
    k_values: list[int],
    num_iterations: int,
    window: int | None,
    in_order_forwarding: bool,
    node_queue_size: int | None,
    packets_per_path: int | None,
    feedback_mode: SRFeedbackMode,
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
                    k, k, float(e1), float(e2), it, jammer_alpha,
                    num_paths, num_hops, rtt, num_packets_to_send, max_iterations,
                    window, in_order_forwarding, node_queue_size, packets_per_path,
                    feedback_mode,
                ))

        results: list[tuple[int, SimulationStats]] = []
        total = len(tasks)
        completed = 0
        for _, result in _execute_tasks(tasks, parallel_workers=parallel_workers, progress_label="sweep_k"):
            (_gk, k, _e1, _e2, it, _alpha, stats) = result
            results.append((k, stats))
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(
                    f"[{completed}/{total}] iter {it}/{num_iterations} k={k:>2} "
                    f"-> tp={stats.normalized_throughput:.4f} "
                    f"delay_mean={stats.inorder_delay_mean:.2f} "
                    f"decoded={stats.num_information_packets_decoded}"
                )
        save_pickle(results, results_file, prefix="sr_jam_sweep_k")

    aggregated = aggregate_by_key(results)
    plot_sweep(
        aggregated,
        x_label="Jammer k (paths blocked per round)",
        title_suffix=(
            f"SR-ARQ [{feedback_mode.name}] JamMpMh sweep_k (H={num_hops}, P={num_paths}, "
            f"alpha={jammer_alpha}, RTT={rtt}, W={window}, eps_template e1={e1} e2={e2}, "
            f"iof={in_order_forwarding}, nqs={node_queue_size})"
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
    num_packets_to_send: int,
    max_iterations: int,
    jammer_k: int,
    alpha_values: list[int],
    num_iterations: int,
    window: int | None,
    in_order_forwarding: bool,
    node_queue_size: int | None,
    packets_per_path: int | None,
    feedback_mode: SRFeedbackMode,
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
            for alpha in alpha_values:
                # group_key = alpha; k is fixed. The result tuple carries alpha at
                # index 5, which is what we aggregate on.
                tasks.append((
                    alpha, jammer_k, float(e1), float(e2), it, alpha,
                    num_paths, num_hops, rtt, num_packets_to_send, max_iterations,
                    window, in_order_forwarding, node_queue_size, packets_per_path,
                    feedback_mode,
                ))

        results: list[tuple[int, SimulationStats]] = []
        total = len(tasks)
        completed = 0
        for _, result in _execute_tasks(tasks, parallel_workers=parallel_workers, progress_label="sweep_alpha"):
            (_gk, _k, _e1, _e2, it, alpha, stats) = result
            results.append((alpha, stats))
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(
                    f"[{completed}/{total}] iter {it}/{num_iterations} alpha={alpha:>2} "
                    f"-> tp={stats.normalized_throughput:.4f} "
                    f"delay_mean={stats.inorder_delay_mean:.2f}"
                )
        save_pickle(results, results_file, prefix="sr_jam_sweep_alpha")

    aggregated = aggregate_by_key(results)
    plot_sweep(
        aggregated,
        x_label="Jammer alpha (jamming round = RTT/alpha)",
        title_suffix=(
            f"SR-ARQ [{feedback_mode.name}] JamMpMh sweep_alpha (H={num_hops}, P={num_paths}, "
            f"k={jammer_k}, RTT={rtt}, W={window}, eps_template e1={e1} e2={e2}, "
            f"iof={in_order_forwarding}, nqs={node_queue_size})"
        ),
        plot_path=plot_file,
    )


def mode_sweep_k_multi_eps(
    *,
    eps_pairs: list[tuple[float, float]],
    num_paths: int,
    num_hops: int,
    rtt: int,
    num_packets_to_send: int,
    max_iterations: int,
    jammer_alpha: int,
    k_values: list[int],
    num_iterations: int,
    window: int | None,
    in_order_forwarding: bool,
    node_queue_size: int | None,
    packets_per_path: int | None,
    feedback_mode: SRFeedbackMode,
    results_file: str,
    plot_file: str,
    load_existing: bool,
    parallel_workers: int = 1,
) -> None:
    """Sweep jammer_k for each (e1, e2) in eps_pairs and overlay all curves on one
    figure (3 subplots: throughput / mean delay / max delay)."""
    if load_existing and os.path.exists(results_file):
        per_eps_results = load_pickle(results_file)
    else:
        per_eps_target_count = num_iterations * len(k_values)
        tasks: list[tuple] = []
        for (e1, e2) in eps_pairs:
            for it in range(1, num_iterations + 1):
                for k in k_values:
                    tasks.append((
                        (float(e1), float(e2)), k, float(e1), float(e2), it, jammer_alpha,
                        num_paths, num_hops, rtt, num_packets_to_send, max_iterations,
                        window, in_order_forwarding, node_queue_size, packets_per_path,
                        feedback_mode,
                    ))

        per_eps_results: dict[tuple[float, float], list[tuple[int, SimulationStats]]] = {
            (float(e1), float(e2)): [] for (e1, e2) in eps_pairs
        }
        per_eps_complete: dict[tuple[float, float], int] = {
            (float(e1), float(e2)): 0 for (e1, e2) in eps_pairs
        }
        total = len(tasks)
        completed = 0
        for _, result in _execute_tasks(tasks, parallel_workers=parallel_workers, progress_label="sweep_k_multi_eps"):
            (group_key, k, e1, e2, it, _alpha, stats) = result
            per_eps_results[group_key].append((k, stats))
            per_eps_complete[group_key] += 1
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(
                    f"[{completed}/{total}] e1={e1:.2f} e2={e2:.2f} "
                    f"iter {it}/{num_iterations} k={k:>2} "
                    f"-> tp={stats.normalized_throughput:.4f} "
                    f"delay_mean={stats.inorder_delay_mean:.2f}"
                )
            if per_eps_complete[group_key] == per_eps_target_count:
                save_pickle(per_eps_results, results_file, prefix="sr_jam_sweep_k_multi_eps")
                print(f"[checkpoint] (e1={e1:.2f}, e2={e2:.2f}) complete -> saved partial pickle")
        save_pickle(per_eps_results, results_file, prefix="sr_jam_sweep_k_multi_eps")

    series: dict[str, dict[int | float, dict]] = {}
    for (e1, e2), results in per_eps_results.items():
        series[f"e1={e1:.2f}, e2={e2:.2f}"] = aggregate_by_key(results)

    plot_sweep_multi(
        series,
        x_label="Jammer k (paths blocked per round)",
        title_suffix=(
            f"SR-ARQ [{feedback_mode.name}] JamMpMh sweep_k_multi_eps "
            f"(H={num_hops}, P={num_paths}, alpha={jammer_alpha}, RTT={rtt}, W={window}, "
            f"iof={in_order_forwarding}, nqs={node_queue_size})"
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
    num_packets_to_send: int,
    max_iterations: int,
    jammer_alpha: int,
    num_iterations: int,
    window: int | None,
    in_order_forwarding: bool,
    node_queue_size: int | None,
    packets_per_path: int | None,
    feedback_mode: SRFeedbackMode,
    results_file: str,
    plot_file: str,
    load_existing: bool,
    parallel_workers: int = 1,
) -> None:
    """Sweep (e1, e2) over the eps_values x eps_values grid for each k, and plot
    one 3D surface per k overlaid on the same (eps1, eps2) plane for throughput /
    mean delay / max delay. Pickle is saved incrementally after each k completes."""
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
                            k, k, float(e1), float(e2), it, jammer_alpha,
                            num_paths, num_hops, rtt, num_packets_to_send, max_iterations,
                            window, in_order_forwarding, node_queue_size, packets_per_path,
                            feedback_mode,
                        ))

        per_k_results: dict[int, list[tuple[float, float, SimulationStats]]] = {
            k: [] for k in k_values
        }
        per_k_complete: dict[int, int] = {k: 0 for k in k_values}
        total = len(tasks)
        completed = 0
        for _, result in _execute_tasks(tasks, parallel_workers=parallel_workers, progress_label="sweep_eps_grid_per_k"):
            (group_key, k, e1, e2, it, _alpha, stats) = result
            per_k_results[group_key].append((e1, e2, stats))
            per_k_complete[group_key] += 1
            completed += 1
            if completed % 50 == 0 or completed == total or completed <= parallel_workers:
                print(
                    f"[{completed}/{total}] k={k:>2} iter {it}/{num_iterations} "
                    f"e1={e1:.1f} e2={e2:.1f} -> "
                    f"tp={stats.normalized_throughput:.4f} "
                    f"delay_mean={stats.inorder_delay_mean:.2f}"
                )
            if per_k_complete[group_key] == per_k_target_count:
                save_pickle(per_k_results, results_file, prefix="sr_jam_sweep_eps_grid_per_k")
                print(f"[checkpoint] k={group_key} complete -> saved partial pickle")
        save_pickle(per_k_results, results_file, prefix="sr_jam_sweep_eps_grid_per_k")

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

    # Min-cut capacity is an AC-RLNC upper bound (SR-ARQ sits well below it); shown
    # only as a reference ceiling on the throughput subplot when NetworkX is present.
    capacity_func = None
    capacity_label = "Min-cut capacity"
    if _mp_mh_min_cut is not None:
        capacity_func = lambda e1, e2: float(_mp_mh_min_cut(e1, e2, num_hops))
        capacity_label = "Min-cut capacity (upper bound; not SR-achievable)"

    plot_per_k_surfaces(
        aggregated_per_k,
        eps_values_e1=eps_sorted,
        eps_values_e2=eps_sorted,
        title_suffix=(
            f"SR-ARQ [{feedback_mode.name}] JamMpMh sweep_eps_grid_per_k "
            f"(H={num_hops}, P={num_paths}, alpha={jammer_alpha}, RTT={rtt}, W={window}, "
            f"iof={in_order_forwarding}, nqs={node_queue_size})"
        ),
        plot_path=plot_file,
        capacity_func=capacity_func,
        capacity_label=capacity_label,
    )


def mode_validate_k0(
    *,
    e1: float,
    e2: float,
    num_paths: int,
    num_hops: int,
    rtt: int,
    num_packets_to_send: int,
    max_iterations: int,
    num_iterations: int,
    window: int | None,
    in_order_forwarding: bool,
    node_queue_size: int | None,
    packets_per_path: int | None,
    feedback_mode: SRFeedbackMode,
    results_file: str,
    load_existing: bool,
) -> None:
    """Run SRJamMpMhNetwork with jammer_k=0 and plain SRMpMhNetwork at the same
    eps. With k=0 the Jammer never blocks anything, so the two should agree up to
    channel-realization noise (this checks the jam wrapper is a faithful superset
    of the plain SR network)."""
    if load_existing and os.path.exists(results_file):
        results = load_pickle(results_file)
    else:
        chain_eps = chain_major_epsilons(e1, e2, num_hops)
        results = {"jam_k0": [], "plain": []}
        total = num_iterations * 2
        sim = 0
        for it in range(1, num_iterations + 1):
            sim += 1
            jam_stats = run_sr_jam_network(
                path_eps_chain_major=chain_eps,
                num_paths=num_paths,
                num_hops=num_hops,
                rtt=rtt,
                num_packets_to_send=num_packets_to_send,
                max_iterations=max_iterations,
                jammer_alpha=2,
                jammer_k=0,
                window=window,
                in_order_forwarding=in_order_forwarding,
                node_queue_size=node_queue_size,
                packets_per_path=packets_per_path,
                feedback_mode=feedback_mode,
            )
            results["jam_k0"].append(jam_stats)
            print(
                f"[{sim}/{total}] iter {it}/{num_iterations} SRJamMpMhNetwork(k=0) "
                f"-> tp={jam_stats.normalized_throughput:.4f} "
                f"delay_mean={jam_stats.inorder_delay_mean:.2f}"
            )
            sim += 1
            plain_stats = run_sr_plain_network(
                path_eps_chain_major=chain_eps,
                num_paths=num_paths,
                num_hops=num_hops,
                rtt=rtt,
                num_packets_to_send=num_packets_to_send,
                max_iterations=max_iterations,
                window=window,
                in_order_forwarding=in_order_forwarding,
                node_queue_size=node_queue_size,
                packets_per_path=packets_per_path,
                feedback_mode=feedback_mode,
            )
            results["plain"].append(plain_stats)
            print(
                f"[{sim}/{total}] iter {it}/{num_iterations} SRMpMhNetwork (plain)  "
                f"-> tp={plain_stats.normalized_throughput:.4f} "
                f"delay_mean={plain_stats.inorder_delay_mean:.2f}"
            )
        save_pickle(results, results_file, prefix="sr_jam_validate_k0")

    def _summarize(label: str, stats_list: list[SimulationStats]) -> None:
        tps = [s.normalized_throughput for s in stats_list]
        dms = [s.inorder_delay_mean for s in stats_list]
        dxs = [s.inorder_delay_max for s in stats_list]
        print(
            f"  {label:<26} "
            f"throughput {statistics.mean(tps):.4f} +- {statistics.pstdev(tps):.4f}  |  "
            f"delay_mean {statistics.mean(dms):.2f} +- {statistics.pstdev(dms):.2f}  |  "
            f"delay_max {statistics.mean(dxs):.2f} +- {statistics.pstdev(dxs):.2f}  "
            f"(n={len(stats_list)})"
        )

    print("\n" + "=" * 70)
    print(f"VALIDATE k=0 [{feedback_mode.name}]: SRJamMpMhNetwork(k=0) vs plain SRMpMhNetwork")
    print("=" * 70)
    _summarize("SRJamMpMhNetwork(k=0)", results["jam_k0"])
    _summarize("SRMpMhNetwork (plain)", results["plain"])
    print(
        "\nNote: with k=0 the Jammer never blocks, so both are the same SR-ARQ "
        "network on the same channel; any gap is only channel-realization noise."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_main() -> None:
    print("=" * 70)
    print(" " * 6 + "SR-ARQ -- SRJamMpMhNetwork simulation driver")
    print("=" * 70)

    # ---- Topology / protocol constants (mirrors jam_mp_mh_simulation.py) --
    NUM_PATHS = 4
    NUM_HOPS = 3
    RTT = 12
    HOP_RTT = RTT // NUM_HOPS
    EPS_E1 = 0.1
    EPS_E2 = 0.2

    # ---- Mode selection --------------------------------------------------
    MODE = "sweep_eps_grid_per_k"  # one of: "sweep_k", "sweep_k_multi_eps", "sweep_eps_grid_per_k", "sweep_alpha", "validate_k0"
    LOAD_EXISTING = False

    # ---- SR feedback mode (exactly one per run) --------------------------
    #   SRFeedbackMode.HBH              -> hop-by-hop per-hop SR-ARQ
    #   SRFeedbackMode.E2E_FORWARD_ONLY -> forward-only relays, slot-based E2E
    #   SRFeedbackMode.E2E_FULL_ARQ     -> full per-hop SR-ARQ relays, seq-based E2E
    #   SRFeedbackMode.E2E_TIMEOUT      -> forward-only relays, ACK-only E2E + source timeout
    SR_FEEDBACK_MODE = SRFeedbackMode.E2E_FORWARD_ONLY
    _FB_TAG = SR_FEEDBACK_MODE.name

    # Window sizing: HBH runs an independent SR-ARQ per hop (per-hop RTT drives the
    # window); the E2E modes run the SR-ARQ end-to-end at the source (window must
    # span a full end-to-end RTT). Same rule as scripts/sr_arq_mpmh_simulation.py.
    SR_WINDOW = 2 * (HOP_RTT - 1)
    # Node forwarding discipline (False = out-of-order relay; True = full SR-ARQ
    # per node / in-order forwarding). Per-relay flow-control buffer
    # (None = unbounded). Equal per-chain new-packet quota (None = use
    # NUM_PACKETS_TO_SEND as the global target).
    IN_ORDER_FORWARDING = False
    NODE_QUEUE_SIZE = None

    NUM_ITERATIONS = 150
    # ---- Stop trigger: pick one (like scripts/sr_arq_mpmh_simulation.py) --------
    #   * MAX_ITERATIONS      -> fixed horizon in slots. Recommended for jam sweeps:
    #                            heavily jammed points are censored at the horizon.
    #   * PACKETS_PER_PATH    -> per-chain quota (global target = PACKETS_PER_PATH * P).
    #   * NUM_PACKETS_TO_SEND -> stop after this many global in-order deliveries.
    # Combining is allowed and matches the network: PACKETS_PER_PATH (if set)
    # overrides NUM_PACKETS_TO_SEND as the delivery target, and MAX_ITERATIONS (if
    # set) additionally caps the run time.
    # WARNING: with MAX_ITERATIONS = None a heavily/fully jammed point (large k /
    # high eps) never reaches its packet target and will NOT terminate; keep a
    # finite MAX_ITERATIONS for any sweep that includes high k.
    MAX_ITERATIONS = None
    PACKETS_PER_PATH = None
    NUM_PACKETS_TO_SEND = 50

    # Parallelization for the sweep modes. 1 = serial. os.cpu_count() // 2 leaves
    # headroom for the OS. Applies to every mode here.
    PARALLEL_WORKERS = max(1, (os.cpu_count() or 2) // 2)

    # Config tag keeps HBH/E2E, different-knob, and different-stop-trigger runs from
    # colliding on disk (mirrors the filename tagging in sr_arq_mpmh_simulation.py).
    # if PACKETS_PER_PATH is not None:
    #     _TRIGGER_TAG = f"ppp{PACKETS_PER_PATH}"
    # elif MAX_ITERATIONS is not None:
    #     _TRIGGER_TAG = f"maxit{MAX_ITERATIONS}"
    # else:
    #     _TRIGGER_TAG = f"npkts{NUM_PACKETS_TO_SEND}"
    _TRIGGER_TAG = f"packets_to_send_{NUM_PACKETS_TO_SEND}_max_iterations_{MAX_ITERATIONS}"
    _CFG_TAG = f"{_FB_TAG}_window_{SR_WINDOW}_RTT_{RTT}_in_order_forwarding_{IN_ORDER_FORWARDING}_node_queue_size_{NODE_QUEUE_SIZE}_{_TRIGGER_TAG}"

    # sweep_k config
    K_VALUES: list[int] = list(range(0, NUM_PATHS * NUM_HOPS + 1))
    SWEEP_K_ALPHA = 2
    K_RESULTS_FILE = f"sr_jam_sweep_k_results_{_CFG_TAG}.pkl"
    K_PLOT_FILE = f"sr_jam_sweep_k_{_CFG_TAG}.png"

    # sweep_k_multi_eps config
    MULTI_EPS_PAIRS: list[tuple[float, float]] = [
        (0.05, 0.05),  # low loss
        (0.1, 0.2),    # paper template
        (0.3, 0.3),    # mid loss
        (0.5, 0.5),    # high loss
    ]
    MULTI_EPS_RESULTS_FILE = f"sr_jam_sweep_k_multi_eps_results_{_CFG_TAG}.pkl"
    MULTI_EPS_PLOT_FILE = f"sr_jam_sweep_k_multi_eps_{_CFG_TAG}.png"

    # sweep_eps_grid_per_k config (same (e1, e2) grid convention as mp_mh)
    EPS_GRID_VALUES: list[float] = [round(float(v), 2) for v in np.arange(0.1, 0.9, 0.1)]
    tot_paths = NUM_HOPS * NUM_PATHS
    # EPS_GRID_K_VALUES: list[int] = [0, int(0.25 * tot_paths), int(0.5 * tot_paths), int(0.75 * tot_paths)]
    EPS_GRID_K_VALUES: list[int] = [9] #!!
    EPS_GRID_RESULTS_FILE = f"sr_jam_sweep_eps_grid_per_k_results_{_CFG_TAG}_k_9.pkl" #!!
    EPS_GRID_PLOT_FILE = f"sr_jam_sweep_eps_grid_per_k_{_CFG_TAG}_k_9.png" #!!

    # sweep_alpha config
    ALPHA_VALUES: list[int] = [1, 2, 3, 4, 6, 12]  # divisors of RTT -> integer RTT/alpha
    SWEEP_ALPHA_K = 2
    ALPHA_RESULTS_FILE = f"sr_jam_sweep_alpha_results_{_CFG_TAG}.pkl"
    ALPHA_PLOT_FILE = f"sr_jam_sweep_alpha_{_CFG_TAG}.png"

    # validate_k0 config
    VALIDATE_RESULTS_FILE = f"sr_jam_validate_k0_results_{_CFG_TAG}.pkl"

    print("\nSimulation parameters:")
    print(f"  - RTT (slots): {RTT}, hop_rtt: {HOP_RTT}")
    print(f"  - Paths P = {NUM_PATHS}, hops H = {NUM_HOPS}")
    print(f"  - Feedback mode: {SR_FEEDBACK_MODE.name}, window: {SR_WINDOW}")
    print(f"  - in_order_forwarding={IN_ORDER_FORWARDING}, node_queue_size={NODE_QUEUE_SIZE}")
    print(f"  - Stop trigger: max_iterations={MAX_ITERATIONS}, "
          f"packets_per_path={PACKETS_PER_PATH}, num_packets_to_send={NUM_PACKETS_TO_SEND}")
    print(f"  - Outer iterations: {NUM_ITERATIONS}")
    print(f"  - eps template: e1={EPS_E1}, e2={EPS_E2}")
    print(f"  - workers: {PARALLEL_WORKERS}")
    print(f"  - MODE: {MODE}\n")

    if MODE == "sweep_k":
        print(f"Running sweep_k with k_values: {K_VALUES}")
        mode_sweep_k(
            e1=EPS_E1, e2=EPS_E2, num_paths=NUM_PATHS, num_hops=NUM_HOPS, rtt=RTT,
            num_packets_to_send=NUM_PACKETS_TO_SEND, max_iterations=MAX_ITERATIONS,
            jammer_alpha=SWEEP_K_ALPHA, k_values=K_VALUES, num_iterations=NUM_ITERATIONS,
            window=SR_WINDOW, in_order_forwarding=IN_ORDER_FORWARDING,
            node_queue_size=NODE_QUEUE_SIZE, packets_per_path=PACKETS_PER_PATH,
            feedback_mode=SR_FEEDBACK_MODE, results_file=K_RESULTS_FILE,
            plot_file=K_PLOT_FILE, load_existing=LOAD_EXISTING,
            parallel_workers=PARALLEL_WORKERS,
        )
    elif MODE == "sweep_k_multi_eps":
        print(f"Running sweep_k_multi_eps with eps_pairs: {MULTI_EPS_PAIRS}")
        mode_sweep_k_multi_eps(
            eps_pairs=MULTI_EPS_PAIRS, num_paths=NUM_PATHS, num_hops=NUM_HOPS, rtt=RTT,
            num_packets_to_send=NUM_PACKETS_TO_SEND, max_iterations=MAX_ITERATIONS,
            jammer_alpha=SWEEP_K_ALPHA, k_values=K_VALUES, num_iterations=NUM_ITERATIONS,
            window=SR_WINDOW, in_order_forwarding=IN_ORDER_FORWARDING,
            node_queue_size=NODE_QUEUE_SIZE, packets_per_path=PACKETS_PER_PATH,
            feedback_mode=SR_FEEDBACK_MODE, results_file=MULTI_EPS_RESULTS_FILE,
            plot_file=MULTI_EPS_PLOT_FILE, load_existing=LOAD_EXISTING,
            parallel_workers=PARALLEL_WORKERS,
        )
    elif MODE == "sweep_eps_grid_per_k":
        print(f"Running sweep_eps_grid_per_k with eps_values: {EPS_GRID_VALUES} and k_values: {EPS_GRID_K_VALUES}")
        mode_sweep_eps_grid_per_k(
            eps_values=EPS_GRID_VALUES, k_values=EPS_GRID_K_VALUES, num_paths=NUM_PATHS,
            num_hops=NUM_HOPS, rtt=RTT, num_packets_to_send=NUM_PACKETS_TO_SEND,
            max_iterations=MAX_ITERATIONS, jammer_alpha=SWEEP_K_ALPHA,
            num_iterations=NUM_ITERATIONS, window=SR_WINDOW,
            in_order_forwarding=IN_ORDER_FORWARDING, node_queue_size=NODE_QUEUE_SIZE,
            packets_per_path=PACKETS_PER_PATH, feedback_mode=SR_FEEDBACK_MODE,
            results_file=EPS_GRID_RESULTS_FILE, plot_file=EPS_GRID_PLOT_FILE,
            load_existing=LOAD_EXISTING, parallel_workers=PARALLEL_WORKERS,
        )
    elif MODE == "sweep_alpha":
        mode_sweep_alpha(
            e1=EPS_E1, e2=EPS_E2, num_paths=NUM_PATHS, num_hops=NUM_HOPS, rtt=RTT,
            num_packets_to_send=NUM_PACKETS_TO_SEND, max_iterations=MAX_ITERATIONS,
            jammer_k=SWEEP_ALPHA_K, alpha_values=ALPHA_VALUES, num_iterations=NUM_ITERATIONS,
            window=SR_WINDOW, in_order_forwarding=IN_ORDER_FORWARDING,
            node_queue_size=NODE_QUEUE_SIZE, packets_per_path=PACKETS_PER_PATH,
            feedback_mode=SR_FEEDBACK_MODE, results_file=ALPHA_RESULTS_FILE,
            plot_file=ALPHA_PLOT_FILE, load_existing=LOAD_EXISTING,
            parallel_workers=PARALLEL_WORKERS,
        )
    elif MODE == "validate_k0":
        mode_validate_k0(
            e1=EPS_E1, e2=EPS_E2, num_paths=NUM_PATHS, num_hops=NUM_HOPS, rtt=RTT,
            num_packets_to_send=NUM_PACKETS_TO_SEND, max_iterations=MAX_ITERATIONS,
            num_iterations=NUM_ITERATIONS, window=SR_WINDOW,
            in_order_forwarding=IN_ORDER_FORWARDING, node_queue_size=NODE_QUEUE_SIZE,
            packets_per_path=PACKETS_PER_PATH, feedback_mode=SR_FEEDBACK_MODE,
            results_file=VALIDATE_RESULTS_FILE, load_existing=LOAD_EXISTING,
        )
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        _run_main()
    except KeyboardInterrupt:
        # Workers are already terminated in _execute_tasks; force-exit to skip the
        # concurrent.futures atexit join, which can otherwise hang on Windows.
        print("\n[ABORTED] Interrupted by user (Ctrl+C).")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
