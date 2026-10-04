"""
Assemble a single per-k jammed-grid result pickle from the split SR-ARQ runs.
=============================================================================

The SR-ARQ jam sweep (sweep_eps_grid_per_k) was run in pieces. This script
stitches those pieces back into one pickle with the same layout as the example
``results/jam_sweep_eps_grid_per_k_results_E2E.pkl`` so that
``scripts/analyze_results_gui.py`` can read and plot it exactly like that file:

    dict[int k] -> list[(eps1, eps2, SimulationStats), ...]

Sources (all window_22, RTT_12, FORWARD_ONLY, packets_to_send_500):
  * k = 0, 3  : taken from ..._k_0_3.pkl  (both are complete 8x8 grids there)
  * k = 6     : taken from ..._k_6.pkl    (complete 8x8 grid there)
  * k = 9     : concatenation of the eight ..._k_9_e1[_N].pkl files, one per
                eps1 column (eps1 = 0.1 .. 0.8, each with eps2 = 0.1 .. 0.8).

Each complete grid is 8 eps1 x 8 eps2 = 64 cells x 150 iterations = 9600 rows.

Run:
    python scripts/assemble_jam_k_grid.py
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import Counter

# --- make repo packages importable so pickled SimulationStats (un)pickles -----
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

_COMMON = ("sr_jam_sweep_eps_grid_per_k_results_E2E_FORWARD_ONLY_window_22_RTT_12_"
           "in_order_forwarding_False_node_queue_size_None_packets_to_send_500_"
           "max_iterations_None")

# k -> (source pickle filename, key in that pickle to copy)
_SINGLE_SOURCES = {
    0: (f"{_COMMON}_k_0_3.pkl", 0),
    3: (f"{_COMMON}_k_0_3.pkl", 3),
    6: (f"{_COMMON}_k_6.pkl", 6),
}

# k = 9 is split across one file per eps1 column; "" .. "_8" map to eps1 0.1 .. 0.8.
_K9_SUFFIXES = ["", "_2", "_3", "_4", "_5", "_6", "_7", "_8"]
_K9_SOURCES = [f"{_COMMON}_k_9_e1{sfx}.pkl" for sfx in _K9_SUFFIXES]

_OUTPUT = os.path.join(_REPO_ROOT, "results", f"{_COMMON}.pkl")

_EXPECTED_EPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_EXPECTED_CELLS = len(_EXPECTED_EPS) ** 2


def _load(fname):
    path = os.path.join(_REPO_ROOT, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _describe(rows):
    """Return (n_rows, n_cells, sorted set of per-cell counts, eps1 set, eps2 set)."""
    cells = Counter((round(float(a), 2), round(float(b), 2)) for a, b, _ in rows)
    e1 = sorted({round(float(a), 2) for a, _b, _s in rows})
    e2 = sorted({round(float(b), 2) for _a, b, _s in rows})
    return len(rows), len(cells), sorted(set(cells.values())), e1, e2


def assemble():
    combined = {}

    # --- k = 0, 3, 6 : copied straight from their source grids ----------------
    for k, (fname, src_key) in _SINGLE_SOURCES.items():
        obj = _load(fname)
        if src_key not in obj:
            raise KeyError(f"key {src_key} not in {fname}")
        combined[k] = list(obj[src_key])

    # --- k = 9 : concatenate the per-eps1 columns -----------------------------
    k9_rows = []
    for fname in _K9_SOURCES:
        obj = _load(fname)
        if 9 not in obj:
            raise KeyError(f"key 9 not in {fname}")
        k9_rows.extend(obj[9])
    combined[9] = k9_rows

    return combined


def validate(combined):
    print("Assembled grid summary")
    print("-" * 70)
    ok = True
    for k in sorted(combined):
        n_rows, n_cells, per_cell, e1, e2 = _describe(combined[k])
        status = "OK"
        if n_cells != _EXPECTED_CELLS or e1 != _EXPECTED_EPS or e2 != _EXPECTED_EPS:
            status = "INCOMPLETE"
            ok = False
        print(f"  k={k}: rows={n_rows:>5}  cells={n_cells:>3}  "
              f"per-cell n={per_cell}  eps1={e1}  status={status}")
        if e1 != _EXPECTED_EPS:
            print(f"        eps1 mismatch -> got {e1}")
        if e2 != _EXPECTED_EPS:
            print(f"        eps2 mismatch -> got {e2}")
    print("-" * 70)
    return ok


def main():
    combined = assemble()
    ok = validate(combined)

    os.makedirs(os.path.dirname(_OUTPUT), exist_ok=True)
    with open(_OUTPUT, "wb") as f:
        pickle.dump(combined, f)
    print(f"Saved -> {os.path.relpath(_OUTPUT, _REPO_ROOT)}")

    # --- prove it round-trips through the GUI's own loader --------------------
    import analyze_results_gui as gui
    records, all_k, errors = gui.aggregate_files([_OUTPUT])
    print(f"GUI loader: series levels={sorted(all_k)}  "
          f"records={len(records)}  errors={errors}")
    kind = gui.series_kind_for_file(_OUTPUT)
    print(f"GUI series kind inferred from filename: {kind!r} (expected 'k')")

    if not ok:
        print("\nWARNING: at least one k level is not a complete 8x8 grid.")


if __name__ == "__main__":
    main()
