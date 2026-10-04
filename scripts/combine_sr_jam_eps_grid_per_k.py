"""
Combine the split SR-ARQ jammed sweep_eps_grid_per_k runs into ONE pickle.

The jammed (e1, e2) grid sweep for the uncoded SR-ARQ baseline was run in pieces
(one SLURM job could not finish every jammer-k level in time), so the completed
data is scattered across several partial pickles. This script stitches the
*complete* per-k blocks back into the single dict layout that the example result
file (results/jam_sweep_eps_grid_per_k_results_E2E.pkl) and the GUI viewer
(scripts/analyze_results_gui.py) expect:

    dict{ k : [ (e1, e2, SimulationStats), ... ] }     # 9600 entries per k
                                                        # = 8x8 eps grid x 150 iters

Assembly (per the runs that actually finished):
    k = 0  <- ..._k_0_3.pkl   key 0   (9600, full grid)
    k = 3  <- ..._k_0_3.pkl   key 3   (9600, full grid)
    k = 6  <- ..._k_6.pkl     key 6   (9600, full grid)
    k = 9  <- concatenation of the 8 per-eps1 slices
             ..._k_9_e1.pkl, ..._k_9_e1_2.pkl, ... ..._k_9_e1_8.pkl
             (each slice = one fixed eps1 in {0.1..0.8} x 8 eps2 x 150 = 1200;
              8 x 1200 = 9600, the full grid)

The partial k=6 / k=9 blocks inside ..._k_0_3.pkl and ..._k_6.pkl are ignored --
only the finished 9600-entry blocks are taken.

Run:
    python scripts/combine_sr_jam_eps_grid_per_k.py

The output filename keeps the "sweep_eps_grid_per_k" token so analyze_results_gui
.py / plot_saved_results.py classify it as a jammer-k surface (series kind 'k').
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import Counter

# --- make repo packages importable so pickled SimulationStats unpickles -------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "mp_mh_network"),
    os.path.join(_REPO_ROOT, "jamming_simulation"),
    os.path.join(_REPO_ROOT, "sr_arq"),
    os.path.join(_REPO_ROOT, "scripts"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Shared stem for every split run (window 22, RTT 12, forward-only E2E, 500 pkts).
_STEM = ("sr_jam_sweep_eps_grid_per_k_results_E2E_FORWARD_ONLY_window_22_RTT_12_"
         "in_order_forwarding_False_node_queue_size_None_packets_to_send_500_"
         "max_iterations_None")

# Where the split pickles live (repo root) and where the combined one is written.
# The combined name is shortened (drops node_queue_size/packets_to_send, uses
# "max_iters") to match the existing results/..._combined_k0_3_k6.pkl convention
# AND to stay under the Windows 260-char MAX_PATH limit; it keeps the
# "sweep_eps_grid_per_k" token so the GUI still classifies it as a jammer-k surface.
_SRC_DIR = _REPO_ROOT
_OUT_DIR = os.path.join(_REPO_ROOT, "results")
_OUT_NAME = ("sr_jam_sweep_eps_grid_per_k_results_E2E_FORWARD_ONLY_window_22_RTT_12_"
             "in_order_forwarding_False_max_iters_None_combined_k0_3_6_9.pkl")
_OUT_PATH = os.path.join(_OUT_DIR, _OUT_NAME)

_EXPECTED_PER_K = 9600          # 8 x 8 eps grid x 150 iterations
_EXPECTED_CELLS = 64            # 8 x 8 eps grid
_EXPECTED_ITERS_PER_CELL = 150


def _load(name):
    path = os.path.join(_SRC_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{name}: expected dict, got {type(obj).__name__}")
    return obj


def _cells(rows):
    """Counter of (round(e1,2), round(e2,2)) -> iteration count for a row list."""
    return Counter((round(float(e1), 2), round(float(e2), 2)) for (e1, e2, _s) in rows)


def _describe(tag, rows):
    c = _cells(rows)
    iters = sorted(set(c.values()))
    print(f"    {tag}: {len(rows)} rows, {len(c)} cells, iters/cell={iters}")


def build():
    combined: dict[int, list] = {}

    # --- k = 0 and k = 3 : the two finished blocks in the k_0_3 pickle ---------
    k03 = _load(f"{_STEM}_k_0_3.pkl")
    for k in (0, 3):
        rows = k03.get(k, [])
        if len(rows) != _EXPECTED_PER_K:
            print(f"[WARN] k={k} from k_0_3 has {len(rows)} rows "
                  f"(expected {_EXPECTED_PER_K})")
        combined[k] = list(rows)
        _describe(f"k={k}  <- {_STEM}_k_0_3.pkl[{k}]", combined[k])

    # --- k = 6 : the finished block in the k_6 pickle -------------------------
    k6 = _load(f"{_STEM}_k_6.pkl")
    rows6 = k6.get(6, [])
    if len(rows6) != _EXPECTED_PER_K:
        print(f"[WARN] k=6 from k_6 has {len(rows6)} rows (expected {_EXPECTED_PER_K})")
    combined[6] = list(rows6)
    _describe(f"k=6  <- {_STEM}_k_6.pkl[6]", combined[6])

    # --- k = 9 : concatenate the 8 per-eps1 slices ---------------------------
    rows9: list = []
    slice_names = [f"{_STEM}_k_9_e1.pkl"] + [
        f"{_STEM}_k_9_e1_{i}.pkl" for i in range(2, 9)
    ]
    for name in slice_names:
        obj = _load(name)
        part = obj.get(9, [])
        e1s = sorted({round(float(e1), 2) for (e1, _e2, _s) in part})
        print(f"    k=9  <- {name}[9]: {len(part)} rows, eps1={e1s}")
        rows9.extend(part)
    combined[9] = rows9
    _describe("k=9  (concatenated 8 slices)", combined[9])

    return combined


def verify(combined):
    """Fail loudly if the stitched layout does not match the example file."""
    ok = True
    print("\n[VERIFY] per-k grid completeness")
    for k in sorted(combined):
        rows = combined[k]
        c = _cells(rows)
        iters = sorted(set(c.values()))
        full = (len(rows) == _EXPECTED_PER_K
                and len(c) == _EXPECTED_CELLS
                and iters == [_EXPECTED_ITERS_PER_CELL])
        status = "OK " if full else "BAD"
        if not full:
            ok = False
        print(f"    [{status}] k={k}: {len(rows)} rows, {len(c)} cells, "
              f"iters/cell={iters}")
    if not ok:
        print("[VERIFY] WARNING: at least one k level is not a full "
              f"{_EXPECTED_CELLS}-cell x {_EXPECTED_ITERS_PER_CELL}-iter grid.")
    return ok


def main():
    print(f"[INFO] source dir: {_SRC_DIR}")
    combined = build()
    verify(combined)
    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(_OUT_PATH, "wb") as f:
        pickle.dump(combined, f)
    print(f"\n[OK] combined pickle written: {_OUT_PATH}")
    print(f"[OK] keys = {sorted(combined)}, "
          f"rows/k = {[len(combined[k]) for k in sorted(combined)]}")


if __name__ == "__main__":
    main()
