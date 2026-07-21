"""SR-ARQ MP-MH simulation (paper Fig. 19, lower graph).

Reproduces the hop-by-hop SR-ARQ MP-MH experiment (H=3, P=4, RTT=12) in the two
settings described in the paper:

  1. "best single path": one global path built from the best (lowest-erasure)
     path of each hop; only one path per hop is active.
  2. "P matched paths": the P global paths obtained by natural matching (each
     hop's rank-p path is matched to the next hop's rank-p path), run
     independently hop-by-hop.

Both settings are just different chain-major epsilon matrices fed into the
existing SRMpMhNetwork (uncoded hop-by-hop SR-ARQ, decoupled per-chain,
sum-of-per-chain-rates throughput). The paper's epsilon matrix (with eps1, eps2
swept over [0.1, 0.8]) is:

    E(e1, e2) = [[e1 , 0.6, 0.3],
                 [0.8, e1 , e1 ],
                 [0.2, e2 , 0.7],
                 [e2 , 0.4, e2 ]]

Run:
    python scripts/sr_arq_mpmh_simulation.py

Set LOAD_EXISTING = True to replot from the pickle without re-running.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sr_arq"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from mp_mh_network.Network import SimulationStats
from sr_arq.SRNetwork import SRMpMhNetwork

# Reuse the generic aggregation / plotting / pickle helpers from the MP driver.
from sr_arq_simulation import aggregate, plot_compare, save_pickle, load_pickle


NUM_PATHS = 4
NUM_HOPS = 3
RTT = 12


# ---------------------------------------------------------------------------
# Epsilon matrix + global-path formation
# ---------------------------------------------------------------------------

def paper_eps_matrix(e1: float, e2: float) -> list[list[float]]:
    """Paper's MP-MH erasure matrix E[path][hop] (H=3, P=4)."""
    return [
        [float(e1), 0.6, 0.3],
        [0.8, float(e1), float(e1)],
        [0.2, float(e2), 0.7],
        [float(e2), 0.4, float(e2)],
    ]


def natural_matched(E: list[list[float]], num_paths: int = NUM_PATHS, num_hops: int = NUM_HOPS) -> list[list[float]]:
    """Natural matching: global path c uses the (c+1)-th lowest-erasure path at
    each hop (rank-to-rank matching). Chain-major result [chain][hop]."""
    return [
        [sorted(E[p][h] for p in range(num_paths))[c] for h in range(num_hops)]
        for c in range(num_paths)
    ]


def best_single_path(E: list[list[float]], num_paths: int = NUM_PATHS, num_hops: int = NUM_HOPS) -> list[list[float]]:
    """Single global path built from the best (lowest-erasure) path of each hop.
    Returns a 1-chain matrix [1][hop]."""
    return [[min(E[p][h] for p in range(num_paths)) for h in range(num_hops)]]


# ---------------------------------------------------------------------------
# Single-sim runners
# ---------------------------------------------------------------------------

def _run_srmpmh(
    matrix: list[list[float]],
    num_paths: int,
    rtt: int,
    num_packets_to_send: int,
    max_iterations: int | None,
    window: int | None,
    in_order_forwarding: bool = False,
) -> SimulationStats:
    net = SRMpMhNetwork(
        path_epsilons=matrix,
        num_paths=num_paths,
        num_hops=len(matrix[0]),
        prop_delay=rtt // 2,
        num_packets_to_send=num_packets_to_send,
        max_iterations=max_iterations,
        window=window,
        in_order_forwarding=in_order_forwarding,
    )
    net.run_sim()
    return net.get_simulation_stats()


def run_best_single(
    e1: float, e2: float, *, rtt: int, num_packets_to_send: int,
    max_iterations: int | None, window: int | None, in_order_forwarding: bool = False,
) -> SimulationStats:
    E = paper_eps_matrix(e1, e2)
    return _run_srmpmh(best_single_path(E), 1, rtt, num_packets_to_send, max_iterations, window, in_order_forwarding)


def run_matched(
    e1: float, e2: float, *, rtt: int, num_packets_to_send: int,
    max_iterations: int | None, window: int | None, in_order_forwarding: bool = False,
) -> SimulationStats:
    E = paper_eps_matrix(e1, e2)
    return _run_srmpmh(natural_matched(E), NUM_PATHS, rtt, num_packets_to_send, max_iterations, window, in_order_forwarding)


# ---------------------------------------------------------------------------
# Parallel task runner (worker must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_one(args: tuple) -> tuple:
    (setting, e1, e2, rtt, num_packets_to_send, max_iterations, window, in_order_forwarding) = args
    if setting == "best":
        stats = run_best_single(
            e1, e2, rtt=rtt, num_packets_to_send=num_packets_to_send,
            max_iterations=max_iterations, window=window, in_order_forwarding=in_order_forwarding,
        )
    else:
        stats = run_matched(
            e1, e2, rtt=rtt, num_packets_to_send=num_packets_to_send,
            max_iterations=max_iterations, window=window, in_order_forwarding=in_order_forwarding,
        )
    return (setting, float(e1), float(e2), stats)


def _execute(tasks: list[tuple], parallel_workers: int):
    total = len(tasks)
    print(f"\nRunning {total} sims with parallel_workers={parallel_workers}\n")
    if parallel_workers is None or parallel_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            yield idx, total, _run_one(task)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as pool:
            fut_to_idx = {pool.submit(_run_one, t): i for i, t in enumerate(tasks, start=1)}
            done = 0
            for fut in as_completed(fut_to_idx):
                done += 1
                yield done, total, fut.result()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_main() -> None:
    print("=" * 70)
    print(" " * 8 + "SR-ARQ MP-MH (paper Fig. 19 lower graph)")
    print("=" * 70)

    # ---- Paper MP-MH setting ---------------------------------------------
    RTT_LOCAL = RTT                                   # 12
    SR_WINDOW = RTT_LOCAL - 1                          # per-chain sliding window
    EPS_VALUES = [round(float(v), 2) for v in np.arange(0.1, 0.9, 0.1)]  # eps1, eps2 in [0.1, 0.8]

    NUM_PACKETS_TO_SEND = 500
    NUM_ITERATIONS = 150
    MAX_ITERATIONS = 40000
    # Node forwarding discipline: False = out-of-order relay (efficient, low delay);
    # True = full SR-ARQ at each node (in-order forwarding, per-hop HOL blocking,
    # higher delay - matches the paper's "full SR-ARQ protocol at each node").
    IN_ORDER_FORWARDING = True
    PARALLEL_WORKERS = max(1, (os.cpu_count() or 2) // 2)

    LOAD_EXISTING = False
    RESULTS_FILE = "sr_arq_mpmh_results.pkl"
    PLOT_FILE = "sr_arq_mpmh_compare.png"

    SETTINGS = ["best", "matched"]

    print("\nParameters:")
    print(f"  P={NUM_PATHS}, H={NUM_HOPS}, RTT={RTT_LOCAL}, window={SR_WINDOW}")
    print(f"  eps1, eps2 grid: {EPS_VALUES}")
    print(f"  packets/run={NUM_PACKETS_TO_SEND}, iterations={NUM_ITERATIONS}, "
          f"in_order_forwarding={IN_ORDER_FORWARDING}, workers={PARALLEL_WORKERS}")

    if LOAD_EXISTING and os.path.exists(RESULTS_FILE):
        results_by_setting = load_pickle(RESULTS_FILE)
    else:
        tasks: list[tuple] = []
        for setting in SETTINGS:
            for _it in range(1, NUM_ITERATIONS + 1):
                for e1 in EPS_VALUES:
                    for e2 in EPS_VALUES:
                        tasks.append((
                            setting, e1, e2, RTT_LOCAL, NUM_PACKETS_TO_SEND,
                            MAX_ITERATIONS, SR_WINDOW, IN_ORDER_FORWARDING,
                        ))

        results_by_setting: dict[str, list[tuple[float, float, SimulationStats]]] = {
            s: [] for s in SETTINGS
        }
        for done, total, result in _execute(tasks, PARALLEL_WORKERS):
            setting, e1, e2, stats = result
            results_by_setting[setting].append((e1, e2, stats))
            if done % 50 == 0 or done == total:
                print(f"[{done}/{total}] {setting:7s} e1={e1:.1f} e2={e2:.1f} "
                      f"tp={stats.normalized_throughput:.3f} dmean={stats.inorder_delay_mean:.1f}")

        save_pickle(results_by_setting, RESULTS_FILE)

    BEST_LABEL = "SR-ARQ best single path"
    MATCHED_LABEL = "SR-ARQ P matched paths"
    series = []
    if results_by_setting.get("best"):
        series.append((BEST_LABEL, aggregate(results_by_setting["best"]), "tab:red"))
    if results_by_setting.get("matched"):
        series.append((MATCHED_LABEL, aggregate(results_by_setting["matched"]), "tab:blue"))

    plot_compare(
        series,
        EPS_VALUES,
        EPS_VALUES,
        title_suffix=(
            f"SR-ARQ MP-MH hop-by-hop (paper Fig. 19 lower); "
            f"H={NUM_HOPS}, P={NUM_PATHS}, RTT={RTT_LOCAL}; {NUM_ITERATIONS} realizations"
        ),
        plot_path=PLOT_FILE,
        std_for=MATCHED_LABEL,
    )
    print("\nDone.")


if __name__ == "__main__":
    _run_main()
