"""
Compare AC-RLNC vs SR-ARQ jammed-grid results (results/*.pkl).

Loads the four sweep_eps_grid_per_k pickles (dict[k -> list[(e1,e2,SimulationStats)]])
and, for the requested (e1,e2) operating points and the common jammer levels k,
reports throughput / mean in-order delay / max in-order delay per protocol+feedback.

Two throughput numbers are reported per cell:
  * tp_report : the pickle's own `normalized_throughput`. NOTE the two protocols
                define this differently -- AC-RLNC = decoded/t (one global ratio);
                SR-ARQ = sum_c(delivered_c / finish_c) (sum of per-chain rates).
                Internally consistent within a protocol (so HBH-vs-E2E is fair) but
                NOT identical across protocols.
  * goodput   : decoded/time_slots recomputed per run then averaged -- a single,
                identical definition for BOTH protocols (fair cross-protocol number).

Writes results/ac_vs_sr_comparison.json for downstream rendering.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("", "mp_mh_network", "jamming_simulation", "sr_arq", "scripts"):
    sys.path.insert(0, os.path.join(_REPO_ROOT, _p))

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")

FILES = {
    "AC_HBH": "jam_sweep_eps_grid_per_k_results_HBH.pkl",
    "AC_E2E": "jam_sweep_eps_grid_per_k_results_E2E.pkl",
    "SR_HBH": "sr_jam_sweep_eps_grid_per_k_results_HBH_window_6_RTT_12_in_order_forwarding_False_node_queue_size_None_packets_to_send_500_max_iterations_4000.pkl",
    "SR_E2E": "sr_jam_sweep_eps_grid_per_k_results_E2E_FORWARD_ONLY_window_6_RTT_12_in_order_forwarding_False_node_queue_size_None_max_iters_None_combined_k0_3_k6.pkl",
}

EPS_POINTS = [(0.1, 0.1), (0.4, 0.4), (0.8, 0.8)]


def _get(stats, *names):
    for n in names:
        if hasattr(stats, n):
            return getattr(stats, n)
    return None


def _stats_block(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "n": int(a.size),
    }


def load_cells() -> dict:
    """Return cells[tag][ "k|e1|e2" ] = {metric blocks}."""
    cells: dict = {}
    all_ks: dict = {}
    for tag, fname in FILES.items():
        path = os.path.join(RESULTS_DIR, fname)
        with open(path, "rb") as f:
            per_k = pickle.load(f)
        ks = sorted(per_k.keys())
        all_ks[tag] = ks

        buck = defaultdict(lambda: {"tp": [], "dm": [], "dx": [], "good": [], "dec": [], "t": []})
        for k in ks:
            for (e1, e2, s) in per_k[k]:
                key = (k, round(float(e1), 2), round(float(e2), 2))
                dec = float(_get(s, "num_information_packets_decoded") or 0.0)
                t = float(_get(s, "time_slots") or 0.0)
                buck[key]["tp"].append(float(_get(s, "normalized_throughput")))
                buck[key]["dm"].append(float(_get(s, "inorder_delay_mean")))
                buck[key]["dx"].append(float(_get(s, "inorder_delay_max")))
                buck[key]["good"].append((dec / t) if t > 0 else 0.0)
                buck[key]["dec"].append(dec)
                buck[key]["t"].append(t)

        cells[tag] = {}
        for (e1, e2) in EPS_POINTS:
            for k in ks:
                b = buck.get((k, e1, e2))
                if not b or not b["tp"]:
                    continue
                cells[tag][f"{k}|{e1}|{e2}"] = {
                    "k": k, "e1": e1, "e2": e2, "n": len(b["tp"]),
                    "tp_report": _stats_block(b["tp"]),
                    "goodput": _stats_block(b["good"]),
                    "delay_mean": _stats_block(b["dm"]),
                    "delay_max": _stats_block(b["dx"]),
                    "decoded_mean": float(np.mean(b["dec"])),
                    "t_mean": float(np.mean(b["t"])),
                    "t_max": float(np.max(b["t"])),
                }
    return {"cells": cells, "all_ks": all_ks}


def _fmt(x, w=8, p=3):
    return f"{x:>{w}.{p}f}"


def print_report(data: dict) -> None:
    cells = data["cells"]

    COMPARISONS = [
        ("1. AC-RLNC HBH  vs  SR-ARQ HBH", "AC_HBH", "SR_HBH"),
        ("2. AC-RLNC E2E  vs  SR-ARQ E2E", "AC_E2E", "SR_E2E"),
        ("3. AC-RLNC HBH  vs  AC-RLNC E2E", "AC_HBH", "AC_E2E"),
        ("4. SR-ARQ HBH  vs  SR-ARQ E2E", "SR_HBH", "SR_E2E"),
    ]

    print("\nCommon k per file:", data["all_ks"])

    for title, A, B in COMPARISONS:
        print("\n" + "=" * 108)
        print(title, f"   [{A}  vs  {B}]")
        print("=" * 108)
        # common k for this pair
        ksA = {int(key.split("|")[0]) for key in cells[A]}
        ksB = {int(key.split("|")[0]) for key in cells[B]}
        ks = sorted(ksA & ksB)
        for k in ks:
            print(f"\n  --- k = {k} ---")
            hdr = (f"  {'eps':>10} | {'goodput A/B (dec/slot)':>26} | "
                   f"{'mean delay A/B':>22} | {'max delay A/B':>22} | tp_report A/B")
            print(hdr)
            print("  " + "-" * 104)
            for (e1, e2) in EPS_POINTS:
                key = f"{k}|{e1}|{e2}"
                ca = cells[A].get(key)
                cb = cells[B].get(key)
                if not ca or not cb:
                    continue
                ga, gb = ca["goodput"]["mean"], cb["goodput"]["mean"]
                da, db = ca["delay_mean"]["mean"], cb["delay_mean"]["mean"]
                xa, xb = ca["delay_max"]["mean"], cb["delay_max"]["mean"]
                ra, rb = ca["tp_report"]["mean"], cb["tp_report"]["mean"]
                note = ""
                if ca["decoded_mean"] < 400 or cb["decoded_mean"] < 400:
                    note = f"  (decoded A={ca['decoded_mean']:.0f} B={cb['decoded_mean']:.0f})"
                print(f"  ({e1},{e2}) | {_fmt(ga)} / {_fmt(gb)}  x{ga/gb if gb else float('nan'):>5.2f} | "
                      f"{_fmt(da,8,1)} / {_fmt(db,8,1)} | {_fmt(xa,8,1)} / {_fmt(xb,8,1)} | "
                      f"{_fmt(ra,6,2)} / {_fmt(rb,6,2)}{note}")


def main() -> None:
    data = load_cells()
    print_report(data)
    out_path = os.path.join(RESULTS_DIR, "ac_vs_sr_comparison.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"\n[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
