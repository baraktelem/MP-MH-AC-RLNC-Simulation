"""SR ARQ MP sanity-check driver (single hop, H=1).

Reproduces the MP setting from the paper (Fig. 11) to sanity-check that the
SR ARQ implementation is on track:

    H = 1, P = 4, RTT = 20, eps_3 = 0.2, eps_4 = 0.8 (fixed),
    eps_1 and eps_2 vary in [0.1, 0.8]; averaged over NUM_ITERATIONS channel
    realizations.

For each (eps_1, eps_2) grid point it runs multipath SR ARQ (SRNetwork) and,
for reference, MP AC-RLNC (MPNetwork) at the same channel, then plots three 3D
surfaces (normalized throughput, mean in-order delay, max in-order delay) with
the two protocols overlaid. SR ARQ should sit clearly below AC-RLNC.

Run:
    python scripts/sr_arq_simulation.py

Set LOAD_EXISTING = True to replot from the pickle without re-running.
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sr_arq"))

from mp_mh_network.Network import MPNetwork, SimulationStats
from sr_arq.SRNetwork import SRNetwork


# ---------------------------------------------------------------------------
# Single-sim runners
# ---------------------------------------------------------------------------

def _path_epsilons(e1: float, e2: float, e3: float, e4: float) -> list[float]:
    return [float(e1), float(e2), float(e3), float(e4)]


def run_sr(
    *,
    path_eps: list[float],
    num_paths: int,
    rtt: int,
    num_packets_to_send: int,
    max_iterations: int | None,
    independent: bool = False,
    debug: bool = False,
) -> SimulationStats:
    net = SRNetwork(
        path_epsilons=path_eps,
        max_iterations=max_iterations,
        num_packets_to_send=num_packets_to_send,
        num_paths=num_paths,
        prop_delay=rtt // 2,
        independent=independent,
        debug=debug,
    )
    net.run_sim()
    return net.get_simulation_stats()


def run_sr_independent_perpath(
    *,
    path_eps: list[float],
    num_paths: int,
    rtt: int,
    num_packets_to_send: int,
    max_iterations: int | None,
    window: int | None = None,
) -> SimulationStats:
    """Fully decoupled per-path SR ARQ: P independent single-path SR-ARQ flows,
    each carrying its OWN stream with its OWN in-order delivery (no global
    reorder across paths). This is the literal 'SR-ARQ used independently on
    each path': a bad path never blocks a good one.

    Aggregation: the total packet budget is split evenly across paths; each path
    runs to completion. Throughput is the SUM of the per-path throughputs
    (delivered_p / t_p) -- the standard aggregate rate of P independent parallel
    connections, which rises as the paths improve (dividing the total by a single
    common finish time instead pins throughput to the slowest fixed path and is
    flat). D_mean/D_max are taken over ALL packets, each measured within its own
    path's in-order stream.
    """
    base, rem = divmod(num_packets_to_send, num_paths)
    all_delays: list[int] = []
    total_delivered = 0
    total_tx = 0
    total_dropped = 0
    total_throughput = 0.0
    for p in range(num_paths):
        n_p = base + (1 if p < rem else 0)
        if n_p == 0:
            continue
        net = SRNetwork(
            path_epsilons=[path_eps[p]],
            max_iterations=max_iterations,
            num_packets_to_send=n_p,
            num_paths=1,
            prop_delay=rtt // 2,
            window=window,
        )
        net.run_sim()
        ftx = net.sender.inforamtion_packets_first_transmission_times
        dec = net.receiver.information_packets_decoding_times
        all_delays.extend(dec[s] - ftx[s] for s in dec)
        total_delivered += len(dec)
        st = net.get_simulation_stats()
        total_throughput += st.normalized_throughput  # per-path delivered_p / t_p
        total_tx += st.num_transmissions
        total_dropped += st.num_transmissions_dropped

    return SimulationStats(
        normalized_throughput=total_throughput,
        inorder_delay_mean=float(np.mean(all_delays)) if all_delays else 0.0,
        inorder_delay_max=int(max(all_delays)) if all_delays else 0,
        num_new_rlnc_packets=total_tx,
        num_fec_packets=0,
        num_fb_fec_packets=0,
        num_transmissions=total_tx,
        num_information_packets_sent=total_delivered,
        num_information_packets_decoded=total_delivered,
        num_transmissions_dropped=total_dropped,
    )


def run_ac_rlnc(
    *,
    path_eps: list[float],
    num_paths: int,
    rtt: int,
    threshold: float,
    o_bar: int,
    num_packets_to_send: int,
    max_iterations: int | None,
    initial_epsilon: float = 0.5,
    debug: bool = False,
) -> SimulationStats:
    net = MPNetwork(
        path_epsilons=path_eps,
        initial_epsilon=initial_epsilon,
        max_iterations=max_iterations,
        num_packets_to_send=num_packets_to_send,
        max_allowed_overlap=o_bar,
        num_paths=num_paths,
        prop_delay=rtt // 2,
        threshold=threshold,
        debug=debug,
    )
    net.run_sim()
    return net.get_simulation_stats()


# ---------------------------------------------------------------------------
# Parallel task runner (worker must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_one_task(args: tuple) -> tuple:
    (proto, e1, e2, e3, e4, it, num_paths, rtt, threshold, o_bar,
     num_packets_to_send, max_iterations, sr_window) = args
    path_eps = _path_epsilons(e1, e2, e3, e4)
    if proto in ("sr", "sr_indep"):
        stats = run_sr(
            path_eps=path_eps,
            num_paths=num_paths,
            rtt=rtt,
            num_packets_to_send=num_packets_to_send,
            max_iterations=max_iterations,
            independent=(proto == "sr_indep"),
        )
    elif proto == "sr_perpath":
        stats = run_sr_independent_perpath(
            path_eps=path_eps,
            num_paths=num_paths,
            rtt=rtt,
            num_packets_to_send=num_packets_to_send,
            max_iterations=max_iterations,
            window=sr_window,
        )
    else:
        stats = run_ac_rlnc(
            path_eps=path_eps,
            num_paths=num_paths,
            rtt=rtt,
            threshold=threshold,
            o_bar=o_bar,
            num_packets_to_send=num_packets_to_send,
            max_iterations=max_iterations,
        )
    return (proto, float(e1), float(e2), it, stats)


def _execute(tasks: list[tuple], parallel_workers: int):
    total = len(tasks)
    print(f"\nRunning {total} sims with parallel_workers={parallel_workers}\n")
    if parallel_workers is None or parallel_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            yield idx, total, _run_one_task(task)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as pool:
            fut_to_idx = {pool.submit(_run_one_task, t): i for i, t in enumerate(tasks, start=1)}
            done = 0
            for fut in as_completed(fut_to_idx):
                done += 1
                yield done, total, fut.result()


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------

def aggregate(results: list[tuple[float, float, SimulationStats]]) -> dict[tuple[float, float], dict]:
    grouped: defaultdict = defaultdict(lambda: {"throughput": [], "delay_mean": [], "delay_max": []})
    for e1, e2, stats in results:
        grouped[(e1, e2)]["throughput"].append(stats.normalized_throughput)
        grouped[(e1, e2)]["delay_mean"].append(stats.inorder_delay_mean)
        grouped[(e1, e2)]["delay_max"].append(stats.inorder_delay_max)
    agg = {}
    for key, v in grouped.items():
        agg[key] = {
            "throughput_mean": float(np.mean(v["throughput"])),
            "throughput_std": float(np.std(v["throughput"])),
            "delay_mean_mean": float(np.mean(v["delay_mean"])),
            "delay_mean_std": float(np.std(v["delay_mean"])),
            "delay_max_mean": float(np.mean(v["delay_max"])),
            "delay_max_std": float(np.std(v["delay_max"])),
            "n": len(v["throughput"]),
        }
    return agg


def _grid(agg: dict, key: str, e1_vals: list[float], e2_vals: list[float]) -> np.ndarray:
    g = np.zeros((len(e1_vals), len(e2_vals)))
    for i, e1 in enumerate(e1_vals):
        for j, e2 in enumerate(e2_vals):
            g[i, j] = agg.get((e1, e2), {}).get(key, 0.0)
    return g


def plot_compare(
    series: list[tuple[str, dict, str]],
    e1_vals: list[float],
    e2_vals: list[float],
    title_suffix: str,
    plot_path: str,
    std_for: str | None = None,
) -> None:
    """series: list of (label, aggregated_dict, color). std_for: the label whose
    std should be drawn as vertical error bars (others drawn as mean surfaces)."""
    EPS1, EPS2 = np.meshgrid(e1_vals, e2_vals)
    metrics = [
        ("throughput", "Normalized Throughput"),
        ("delay_mean", "Mean In-Order Delay"),
        ("delay_max", "Max In-Order Delay"),
    ]
    fig = plt.figure(figsize=(20, 6))
    for idx, (metric, label) in enumerate(metrics):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        max_z = 0.0
        legend = []
        for series_label, agg, color in series:
            mean_g = _grid(agg, f"{metric}_mean", e1_vals, e2_vals)
            ax.plot_surface(EPS1, EPS2, mean_g.T, color=color, edgecolor="none", alpha=0.55)
            max_z = max(max_z, float(np.max(mean_g)))
            if std_for == series_label:
                std_g = _grid(agg, f"{metric}_std", e1_vals, e2_vals)
                for i, e1 in enumerate(e1_vals):
                    for j, e2 in enumerate(e2_vals):
                        m, s = mean_g[i, j], std_g[i, j]
                        ax.plot([e1, e1], [e2, e2], [m - s, m + s],
                                color="black", linewidth=0.7, alpha=0.5)
            legend.append(Patch(color=color, label=series_label, alpha=0.55))

        ax.set_xlabel("eps_1")
        ax.set_ylabel("eps_2")
        ax.set_zlabel(label)
        ax.set_zlim(0, max(0.1, max_z * 1.1))
        ax.set_title(label)
        ax.view_init(elev=20, azim=45)
        ax.legend(handles=legend, fontsize=8, loc="upper left")

    fig.suptitle(title_suffix, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Plot saved to: {plot_path}")
    plt.show()


def save_pickle(obj, filename: str) -> str:
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"[OK] Results saved to: {filename}")
    return filename


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    print(f"[OK] Results loaded from: {filename}")
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_main() -> None:
    print("=" * 70)
    print(" " * 10 + "SR ARQ vs MP AC-RLNC -- MP sanity check (H=1)")
    print("=" * 70)

    # ---- Paper Fig. 11 MP setting ----------------------------------------
    NUM_PATHS = 4
    RTT = 20
    THRESHOLD = 0.0
    O_BAR = 2 * NUM_PATHS * (RTT - 1)        # ō = 2w for AC-RLNC
    EPS3 = 0.2                                # eps_31 (fixed)
    EPS4 = 0.8                                # eps_41 (fixed)
    EPS_VALUES = [round(float(v), 2) for v in np.arange(0.1, 0.9, 0.1)]  # eps_1, eps_2 in [0.1, 0.8]

    # ---- Run controls (paper uses 150 realizations; lower for a quick check)
    NUM_PACKETS_TO_SEND = 500
    NUM_ITERATIONS = 150
    MAX_ITERATIONS = 20000
    RUN_AC_RLNC = False
    # Per-path sliding-window size for the sr_perpath (decoupled) SR-ARQ. None =
    # unbounded (sender front-loads, throughput approaches capacity). A finite w
    # (e.g. RTT-1) throttles throughput below the link rate, like real SR-ARQ.
    SR_WINDOW = RTT - 1
    PARALLEL_WORKERS = max(1, (os.cpu_count() or 2) // 2)

    LOAD_EXISTING = False
    RESULTS_FILE = "sr_arq_mp_results.pkl"
    PLOT_FILE = "sr_arq_mp_compare.png"

    print("\nParameters:")
    print(f"  P={NUM_PATHS}, RTT={RTT}, eps_3={EPS3}, eps_4={EPS4}")
    print(f"  eps_1, eps_2 grid: {EPS_VALUES}")
    print(f"  packets/run={NUM_PACKETS_TO_SEND}, iterations={NUM_ITERATIONS}, "
          f"compare_ac_rlnc={RUN_AC_RLNC}, sr_window={SR_WINDOW}, workers={PARALLEL_WORKERS}")

    if LOAD_EXISTING and os.path.exists(RESULTS_FILE):
        bundle = load_pickle(RESULTS_FILE)
        results_by_proto = bundle
    else:
        # protos = ["sr", "sr_indep", "sr_perpath"] + (["ac"] if RUN_AC_RLNC else [])
        protos = ["sr_perpath"]
        tasks: list[tuple] = []
        for proto in protos:
            for it in range(1, NUM_ITERATIONS + 1):
                for e1 in EPS_VALUES:
                    for e2 in EPS_VALUES:
                        tasks.append((
                            proto, e1, e2, EPS3, EPS4, it, NUM_PATHS, RTT,
                            THRESHOLD, O_BAR, NUM_PACKETS_TO_SEND, MAX_ITERATIONS,
                            SR_WINDOW,
                        ))

        results_by_proto: dict[str, list[tuple[float, float, SimulationStats]]] = {
            p: [] for p in protos
        }
        for done, total, result in _execute(tasks, PARALLEL_WORKERS):
            proto, e1, e2, it, stats = result
            results_by_proto[proto].append((e1, e2, stats))
            if done % 50 == 0 or done == total:
                print(f"[{done}/{total}] {proto:8s} e1={e1:.1f} e2={e2:.1f} "
                      f"tp={stats.normalized_throughput:.3f} dmean={stats.inorder_delay_mean:.1f}")

        save_pickle(results_by_proto, RESULTS_FILE)

    # Build plot series: shared multipath SR (blue), coupled round-robin SR
    # (green), decoupled per-path SR = paper's SP:SR-ARQ baseline (red, with std
    # bars), MP AC-RLNC (orange) if present.
    PERPATH_LABEL = "SP SR-ARQ (per-path independent)"
    series = []
    if results_by_proto.get("sr"):
        series.append(("SR ARQ (shared multipath)", aggregate(results_by_proto["sr"]), "tab:blue"))
    if results_by_proto.get("sr_indep"):
        series.append(("SR ARQ (coupled round-robin)", aggregate(results_by_proto["sr_indep"]), "tab:green"))
    if results_by_proto.get("sr_perpath"):
        series.append((PERPATH_LABEL, aggregate(results_by_proto["sr_perpath"]), "tab:red"))
    if results_by_proto.get("ac"):
        series.append(("MP AC-RLNC", aggregate(results_by_proto["ac"]), "tab:orange"))

    plot_compare(
        series,
        EPS_VALUES,
        EPS_VALUES,
        title_suffix=(
            f"MP sanity check (H=1, P={NUM_PATHS}, RTT={RTT}, eps_3={EPS3}, eps_4={EPS4}); "
            f"SR ARQ (shared / coupled / per-path) vs MP AC-RLNC; {NUM_ITERATIONS} realizations"
        ),
        plot_path=PLOT_FILE,
        std_for=PERPATH_LABEL,
    )
    print("\nDone.")


if __name__ == "__main__":
    _run_main()
