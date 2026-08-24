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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sr_arq"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from mp_mh_network.Network import SimulationStats
from sr_arq.SRNetwork import SRMpMhNetwork
from sr_arq.sr_feedback import SRFeedbackMode

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
    num_packets_to_send: int | None,
    max_iterations: int | None,
    window: int | None,
    in_order_forwarding: bool = False,
    packets_per_path: int | None = None,
    node_queue_size: int | None = None,
    feedback_mode: SRFeedbackMode = SRFeedbackMode.HBH,
) -> SimulationStats:
    net = SRMpMhNetwork(
        path_epsilons=matrix,
        num_paths=num_paths,
        num_hops=len(matrix[0]),
        global_prop_delay=rtt // 2,  # rtt is the end-to-end RTT; split across hops
        num_packets_to_send=num_packets_to_send,
        max_iterations=max_iterations,
        window=window,
        in_order_forwarding=in_order_forwarding,
        node_queue_size=node_queue_size,
        packets_per_path=packets_per_path,
        feedback_mode=feedback_mode,
    )
    net.run_sim()
    return net.get_simulation_stats()


def run_best_single(
    e1: float, e2: float, *, rtt: int, num_packets_to_send: int | None,
    max_iterations: int | None, window: int | None, in_order_forwarding: bool = False,
    packets_per_path: int | None = None, node_queue_size: int | None = None,
    feedback_mode: SRFeedbackMode = SRFeedbackMode.HBH,
) -> SimulationStats:
    E = paper_eps_matrix(e1, e2)
    return _run_srmpmh(
        best_single_path(E), 1, rtt, num_packets_to_send, max_iterations, window,
        in_order_forwarding, packets_per_path=packets_per_path, node_queue_size=node_queue_size,
        feedback_mode=feedback_mode,
    )


def run_matched(
    e1: float, e2: float, *, rtt: int, num_packets_to_send: int | None,
    max_iterations: int | None, window: int | None, in_order_forwarding: bool = False,
    packets_per_path: int | None = None, node_queue_size: int | None = None,
    feedback_mode: SRFeedbackMode = SRFeedbackMode.HBH,
) -> SimulationStats:
    E = paper_eps_matrix(e1, e2)
    return _run_srmpmh(
        natural_matched(E), NUM_PATHS, rtt, num_packets_to_send, max_iterations, window,
        in_order_forwarding, packets_per_path=packets_per_path, node_queue_size=node_queue_size,
        feedback_mode=feedback_mode,
    )


# ---------------------------------------------------------------------------
# Parallel task runner (worker must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_one(args: tuple) -> tuple:
    (setting, e1, e2, rtt, num_packets_to_send, max_iterations, window,
     in_order_forwarding, packets_per_path, node_queue_size, feedback_mode) = args
    if setting == "best":
        stats = run_best_single(
            e1, e2, rtt=rtt, num_packets_to_send=num_packets_to_send,
            max_iterations=max_iterations, window=window,
            in_order_forwarding=in_order_forwarding,
            packets_per_path=packets_per_path, node_queue_size=node_queue_size,
            feedback_mode=feedback_mode,
        )
    else:
        stats = run_matched(
            e1, e2, rtt=rtt, num_packets_to_send=num_packets_to_send,
            max_iterations=max_iterations, window=window,
            in_order_forwarding=in_order_forwarding,
            packets_per_path=packets_per_path, node_queue_size=node_queue_size,
            feedback_mode=feedback_mode,
        )
    return (setting, float(e1), float(e2), stats)


def _execute(tasks: list[tuple], parallel_workers: int):
    total = len(tasks)
    print(f"\nRunning {total} sims with parallel_workers={parallel_workers}\n")
    if parallel_workers is None or parallel_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            yield idx, total, _run_one(task)
        return

    # Manage the pool manually so Ctrl+C stops promptly. Using
    # `with ProcessPoolExecutor(...)` blocks in shutdown(wait=True) on exit, and
    # an untimed as_completed()/Event.wait() suppresses KeyboardInterrupt
    # delivery on Windows. Polling with a short timeout lets the main thread
    # raise the pending interrupt and reach the handler, which cancels queued
    # work and terminates the worker processes.
    pool = ProcessPoolExecutor(max_workers=parallel_workers)
    futures: list = []
    try:
        futures = [pool.submit(_run_one, t) for t in tasks]
        pending = set(futures)
        done = 0
        while pending:
            finished, pending = wait(
                pending, timeout=0.5, return_when=FIRST_COMPLETED
            )
            for fut in finished:
                done += 1
                yield done, total, fut.result()
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
# Main
# ---------------------------------------------------------------------------

def _run_main() -> None:
    print("=" * 70)
    print(" " * 8 + "SR-ARQ MP-MH (paper Fig. 19 lower graph)")
    print("=" * 70)

    # ---- Paper MP-MH setting ---------------------------------------------
    EPS_VALUES = [round(float(v), 2) for v in np.arange(0.1, 0.9, 0.1)]  # eps1, eps2 in [0.1, 0.8]

    HOP_RTT = RTT // NUM_HOPS                                   
    # RTT is the end-to-end (global) round-trip; SRMpMhNetwork/MhNetwork splits it
    # across the H hops. The per-hop RTT (HOP_RTT = RTT / H) drives the HBH window.

    # Feedback / relay mode:
    #   SRFeedbackMode.HBH              -> hop-by-hop (paper Fig. 19 bottom; the default)
    #   SRFeedbackMode.E2E_FORWARD_ONLY -> forward-only relays, slot-based E2E (paper Fig. 19 top)
    #   SRFeedbackMode.E2E_FULL_ARQ     -> full per-hop SR-ARQ relays, seq-based E2E (mp_mh-style)
    SR_FEEDBACK_MODE = SRFeedbackMode.E2E_FORWARD_ONLY
    _FB_TAG = SR_FEEDBACK_MODE.name  # keeps HBH / E2E outputs from colliding

    # Window sizing. HBH runs an independent SR-ARQ per hop, so the per-hop RTT
    # drives the window. Both E2E modes run the SR-ARQ end-to-end at the source, so
    # the per-chain window must span a full end-to-end RTT; a per-hop window would
    # stall each chain for a full RTT per end-to-end ACK and crush throughput.
    # if SR_FEEDBACK_MODE.is_e2e():
    #     SR_WINDOW = 2 * (RTT - 1)
    # else:
    #     SR_WINDOW = 2 * (HOP_RTT - 1)
    SR_WINDOW = 2 * (HOP_RTT - 1)

    NUM_ITERATIONS = 150
    # Equal new-packet quota per chain (None = unlimited). When set, each chain
    # admits exactly this many new seqs; global delivery target becomes P * N
    # (or 1 * N for the best single-path setting).
    PACKETS_PER_PATH = None
    # Optional hard time stop (None = run until all quota packets are delivered).
    MAX_ITERATIONS = 2000
    # Legacy global packet target; ignored when PACKETS_PER_PATH is set.
    NUM_PACKETS_TO_SEND = None
    # Node forwarding discipline: False = out-of-order relay (efficient, low delay);
    # True = full SR-ARQ at each node (in-order forwarding, per-hop HOL blocking,
    # higher delay - matches the paper's "full SR-ARQ protocol at each node").
    IN_ORDER_FORWARDING = True
    # Per-relay flow-control buffer: max received-but-not-forwarded seqs a node
    # may hold before it applies backpressure (refuses new arrivals so the
    # upstream retransmits later). None = unbounded (no backpressure); a finite
    # value bounds the bottleneck queue and makes the in-order delay stationary.
    NODE_QUEUE_SIZE = 256

    PARALLEL_WORKERS = max(1, (os.cpu_count() or 2) // 2)

    LOAD_EXISTING = False
    # Keep the complete-cohort (post-horizon drain) experiment separate from
    # the earlier hard-cutoff results, whose delays were right-censored.
    if PACKETS_PER_PATH is not None:
        RESULTS_FILE = f"sr_arq_mpmh_results_{_FB_TAG}_packets_per_path_{PACKETS_PER_PATH}_window_{SR_WINDOW}_RTT_{RTT}_in_order_forwarding_{IN_ORDER_FORWARDING}_node_queue_size_{NODE_QUEUE_SIZE}.pkl"
        PLOT_FILE = f"sr_arq_mpmh_compare_{_FB_TAG}_packets_per_path_{PACKETS_PER_PATH}_window_{SR_WINDOW}_RTT_{RTT}_in_order_forwarding_{IN_ORDER_FORWARDING}_node_queue_size_{NODE_QUEUE_SIZE}.png"
    elif MAX_ITERATIONS is not None:
        RESULTS_FILE = f"sr_arq_mpmh_results_{_FB_TAG}_max_iterations_{MAX_ITERATIONS}_window_{SR_WINDOW}_RTT_{RTT}_in_order_forwarding_{IN_ORDER_FORWARDING}_node_queue_size_{NODE_QUEUE_SIZE}.pkl"
        PLOT_FILE = f"sr_arq_mpmh_compare_{_FB_TAG}_max_iterations_{MAX_ITERATIONS}_window_{SR_WINDOW}_RTT_{RTT}_in_order_forwarding_{IN_ORDER_FORWARDING}_node_queue_size_{NODE_QUEUE_SIZE}.png"
    else:
        RESULTS_FILE = f"sr_arq_mpmh_results_{_FB_TAG}_num_packets_to_send_{NUM_PACKETS_TO_SEND}_window_{SR_WINDOW}_RTT_{RTT}_in_order_forwarding_{IN_ORDER_FORWARDING}_node_queue_size_{NODE_QUEUE_SIZE}.pkl"
        PLOT_FILE = f"sr_arq_mpmh_compare_{_FB_TAG}_num_packets_to_send_{NUM_PACKETS_TO_SEND}_window_{SR_WINDOW}_RTT_{RTT}_in_order_forwarding_{IN_ORDER_FORWARDING}_node_queue_size_{NODE_QUEUE_SIZE}.png"

    SETTINGS = ["best", "matched"] if not SR_FEEDBACK_MODE.is_e2e() else ["matched"]

    print("\nParameters:")
    print(f"  feedback_mode={SR_FEEDBACK_MODE.name}")
    print(f"  P={NUM_PATHS}, H={NUM_HOPS}, RTT={RTT} (hop_rtt={HOP_RTT}), window={SR_WINDOW}, node_queue_size={NODE_QUEUE_SIZE}")
    print(f"  eps1, eps2 grid: {EPS_VALUES}")
    print(f"  packets_per_path={PACKETS_PER_PATH}, max_iterations={MAX_ITERATIONS}")
    print(f"  num_packets_to_send={NUM_PACKETS_TO_SEND} (legacy; unused if packets_per_path set)")
    print(f"  iterations={NUM_ITERATIONS}, in_order_forwarding={IN_ORDER_FORWARDING}, "
          f"workers={PARALLEL_WORKERS}")
    print(f"  node queue size={NODE_QUEUE_SIZE}")
    print(f"\nresults file:\n{RESULTS_FILE}")
    print(f"\nplot file:\n{PLOT_FILE}")
    print("(saved only when all simulations are done)")

    if LOAD_EXISTING and os.path.exists(RESULTS_FILE):
        results_by_setting = load_pickle(RESULTS_FILE)
    else:
        tasks: list[tuple] = []
        for setting in SETTINGS:
            for _it in range(1, NUM_ITERATIONS + 1):
                for e1 in EPS_VALUES:
                    for e2 in EPS_VALUES:
                        tasks.append((
                            setting, e1, e2, RTT, NUM_PACKETS_TO_SEND,
                            MAX_ITERATIONS, SR_WINDOW, IN_ORDER_FORWARDING,
                            PACKETS_PER_PATH, NODE_QUEUE_SIZE, SR_FEEDBACK_MODE,
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

    BEST_LABEL = "SR-ARQ only the best path per hop"
    MATCHED_LABEL = "SR-ARQ all paths with natural matching"
    series = []
    if results_by_setting.get("best"):
        series.append((BEST_LABEL, aggregate(results_by_setting["best"]), "tab:red"))
    if results_by_setting.get("matched"):
        series.append((MATCHED_LABEL, aggregate(results_by_setting["matched"]), "tab:blue"))

    max_iterations_label = (
        f"max iterations={MAX_ITERATIONS}"
    )
    quota_label = (
        f"packets/path={PACKETS_PER_PATH}"
        if PACKETS_PER_PATH is not None
        else f"num packets={NUM_PACKETS_TO_SEND}"
    )
    plot_compare(
        series,
        EPS_VALUES,
        EPS_VALUES,
        title_suffix=(
            f"SR-ARQ MP-MH [{_FB_TAG}]; "
            f"H={NUM_HOPS}, P={NUM_PATHS}, RTT={RTT}, W={SR_WINDOW}; "
            f"{quota_label}, {max_iterations_label}; {NUM_ITERATIONS} realizations; "
            f"in order forwarding={IN_ORDER_FORWARDING}; "
            f"node queue size={NODE_QUEUE_SIZE}"
        ),
        plot_path=PLOT_FILE,
        std_for=MATCHED_LABEL,
    )
    print("\nDone.")


if __name__ == "__main__":
    try:
        _run_main()
    except KeyboardInterrupt:
        # Workers are already terminated in _execute; force-exit to skip the
        # concurrent.futures atexit join, which can otherwise hang on Windows.
        print("\n[ABORTED] Interrupted by user (Ctrl+C).")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
