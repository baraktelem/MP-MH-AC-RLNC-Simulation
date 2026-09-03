"""
Results Analyzer GUI
====================

A small Tkinter desktop app for eyeballing jammed-grid simulation results.

Give it:
  * a list of result pickles (the sweep_eps_grid_per_k files:
    dict[k -> list[(e1, e2, SimulationStats)]]), and
  * a list of (eps1, eps2) operating points,

and it shows ONE sortable table per jammer level k (as Notebook tabs) with
throughput, mean in-order delay and max in-order delay (mean over the stored
iterations) for every (file x eps) row.

Run:
    python scripts/analyze_results_gui.py
    python scripts/analyze_results_gui.py results/a.pkl results/b.pkl --eps "(0.1,0.1),(0.4,0.4),(0.8,0.8)"

Notes on throughput: the table's "Throughput" column is the pickle's stored
`normalized_throughput`, which AC-RLNC and SR-ARQ define differently (AC = one
global decoded/t ratio; SR = sum of per-chain rates). Tick "Show goodput/decoded"
to add `Goodput = decoded/time_slots`, a single identical definition for every
protocol (a fair cross-protocol number).
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    _HAVE_MPL = True
except Exception:  # matplotlib missing -> tables still work, graphs are skipped
    _HAVE_MPL = False

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

_RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
_DEFAULT_EPS = "(0.1,0.1), (0.4,0.4), (0.8,0.8)"


# ---------------------------------------------------------------------------
# Data loading + aggregation
# (based on scripts/_cmp_extract.py and scripts/analyze_ac_vs_sr.py::_stats_block)
# ---------------------------------------------------------------------------

def _get(stats, *names):
    """First present attribute among names, else None."""
    for n in names:
        if hasattr(stats, n):
            return getattr(stats, n)
    return None


def load_file(path):
    """Load one results pickle into {k_label: [(e1, e2, stats), ...]}.

    Tolerant to the three shapes plot_saved_results.py recognises:
      * dict with int keys   -> per-k grid (the jam sweep_eps_grid_per_k files)
      * dict with str keys   -> multi-series; the series name is used as the label
      * list                 -> a single flat group, labelled "-"
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        groups = {k: v for k, v in obj.items() if isinstance(v, list)}
        if not groups:
            raise ValueError("dict pickle has no list values (unrecognised layout)")
        return groups
    if isinstance(obj, list):
        return {"-": obj}
    raise ValueError(f"unsupported pickle top-level type: {type(obj).__name__}")


def aggregate_files(paths):
    """Return (records, all_k, errors).

    records[(path, k, e1, e2)] = {tp_mean, tp_std, dm_mean, dm_std, dx_mean,
                                  dx_std, good_mean, dec_mean, t_mean, n}
    all_k   = set of every k label seen
    errors  = {path: message} for files that failed to load
    """
    records = {}
    all_k = set()
    errors = {}
    for path in paths:
        try:
            groups = load_file(path)
        except Exception as exc:  # surface any load failure to the UI
            errors[path] = str(exc)
            continue
        for k, rows in groups.items():
            all_k.add(k)
            buck = defaultdict(lambda: {"tp": [], "dm": [], "dx": [], "good": [], "dec": [], "t": []})
            for item in rows:
                try:
                    e1, e2, s = item
                except (TypeError, ValueError):
                    continue
                key = (round(float(e1), 2), round(float(e2), 2))
                dec = float(_get(s, "num_information_packets_decoded") or 0.0)
                t = float(_get(s, "time_slots") or 0.0)
                buck[key]["tp"].append(float(_get(s, "normalized_throughput") or 0.0))
                buck[key]["dm"].append(float(_get(s, "inorder_delay_mean") or 0.0))
                buck[key]["dx"].append(float(_get(s, "inorder_delay_max") or 0.0))
                buck[key]["good"].append((dec / t) if t > 0 else 0.0)
                buck[key]["dec"].append(dec)
                buck[key]["t"].append(t)
            for (e1, e2), b in buck.items():
                tp = np.asarray(b["tp"], float)
                dm = np.asarray(b["dm"], float)
                dx = np.asarray(b["dx"], float)
                good = np.asarray(b["good"], float)
                records[(path, k, e1, e2)] = {
                    "tp_mean": float(tp.mean()), "tp_std": float(tp.std()),
                    "dm_mean": float(dm.mean()), "dm_std": float(dm.std()),
                    "dx_mean": float(dx.mean()), "dx_std": float(dx.std()),
                    "good_mean": float(good.mean()),
                    "dec_mean": float(np.asarray(b["dec"], float).mean()),
                    "t_mean": float(np.asarray(b["t"], float).mean()),
                    "n": int(tp.size),
                }
    return records, all_k, errors


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_eps(text):
    """Parse '(0.1,0.1), (0.4,0.4); 0.8,0.8' -> [(0.1,0.1),(0.4,0.4),(0.8,0.8)].

    Grabs every number in order and pairs them up; a dangling odd number is
    ignored. Values are rounded to 2 decimals to match the grid keys.
    """
    nums = [float(x) for x in _NUM_RE.findall(text or "")]
    return [(round(nums[i], 2), round(nums[i + 1], 2)) for i in range(0, len(nums) - 1, 2)]


def _k_sort_key(k):
    """Sort int/float k values numerically first, then any string labels."""
    return (0, k) if isinstance(k, (int, float)) else (1, str(k))


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def fmt_tp(x):
    return f"{x:.4f}" if x < 0.1 else f"{x:.3f}"


def fmt_good(x):
    if x <= 0:
        return "0"
    if x < 0.01:
        return f"{x:.6f}"
    if x < 0.1:
        return f"{x:.4f}"
    return f"{x:.3f}"


def fmt_delay(x):
    return f"{x:,.0f}" if x >= 1000 else f"{x:.1f}"


def fmt_int(x):
    return f"{x:,.0f}"


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class AnalyzerApp(tk.Tk):
    def __init__(self, initial_files=None, initial_eps=None):
        super().__init__()
        self.title("Results Analyzer")
        self.geometry("1120x700")

        self.files = []          # list[str] of absolute pickle paths
        self.records = {}        # (path, k, e1, e2) -> metrics dict
        self.all_k = set()
        self._built = []         # (k, path, e1, e2) rows currently shown, in order
        self._trees = {}         # k -> Treeview
        self._load_errors = {}   # path -> error message (from last aggregation)
        self._agg_files_key = None  # cache key: file set the records were built from

        self.all_eps_var = tk.BooleanVar(value=False)
        self.show_std_var = tk.BooleanVar(value=False)
        self.show_good_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(
            value="Add .pkl files, set (eps1,eps2) points, then Build tables."
        )

        self._build_widgets()

        for p in initial_files or []:
            self._add_path(p)
        if initial_eps:
            self.eps_entry.delete(0, tk.END)
            self.eps_entry.insert(0, initial_eps)
        if self.files:
            self.build_tables()

    # ---- widget construction ---------------------------------------------
    def _build_widgets(self):
        pad = {"padx": 6, "pady": 4}

        files_frame = ttk.LabelFrame(self, text="Pickle files")
        files_frame.pack(fill="x", **pad)
        self.file_list = tk.Listbox(files_frame, height=4, selectmode="extended", activestyle="none")
        self.file_list.pack(side="left", fill="x", expand=True, padx=(6, 0), pady=6)
        fsb = ttk.Scrollbar(files_frame, orient="vertical", command=self.file_list.yview)
        fsb.pack(side="left", fill="y", pady=6)
        self.file_list.configure(yscrollcommand=fsb.set)
        btns = ttk.Frame(files_frame)
        btns.pack(side="left", fill="y", padx=6, pady=6)
        ttk.Button(btns, text="Add...", command=self.add_files).pack(fill="x")
        ttk.Button(btns, text="Remove", command=self.remove_selected).pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Clear", command=self.clear_files).pack(fill="x", pady=(4, 0))

        eps_frame = ttk.Frame(self)
        eps_frame.pack(fill="x", **pad)
        ttk.Label(eps_frame, text="(eps1, eps2) points:").pack(side="left")
        self.eps_entry = ttk.Entry(eps_frame)
        self.eps_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.eps_entry.insert(0, _DEFAULT_EPS)
        ttk.Checkbutton(eps_frame, text="All eps in files", variable=self.all_eps_var).pack(side="left")

        opts_frame = ttk.Frame(self)
        opts_frame.pack(fill="x", **pad)
        ttk.Checkbutton(
            opts_frame, text="Show std", variable=self.show_std_var,
            command=self._on_option_toggle,
        ).pack(side="left")
        ttk.Checkbutton(
            opts_frame, text="Show goodput/decoded", variable=self.show_good_var,
            command=self._on_option_toggle,
        ).pack(side="left", padx=(12, 0))
        ttk.Button(opts_frame, text="Build tables", command=self.build_tables).pack(side="right")
        ttk.Button(opts_frame, text="Export CSV", command=self.export_csv).pack(side="right", padx=(0, 6))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            fill="x", side="bottom"
        )

    # ---- file list handling ----------------------------------------------
    def _add_path(self, path):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            messagebox.showerror("File not found", path)
            return
        if path in self.files:
            return
        self.files.append(path)
        self.file_list.insert(tk.END, os.path.basename(path))

    def add_files(self):
        initial = _RESULTS_DIR if os.path.isdir(_RESULTS_DIR) else _REPO_ROOT
        for p in filedialog.askopenfilenames(
            title="Select result pickle(s)",
            initialdir=initial,
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        ):
            self._add_path(p)

    def remove_selected(self):
        for idx in sorted(self.file_list.curselection(), reverse=True):
            self.file_list.delete(idx)
            del self.files[idx]

    def clear_files(self):
        self.file_list.delete(0, tk.END)
        self.files.clear()

    # ---- table columns ----------------------------------------------------
    def _columns(self):
        """Return list of (colid, heading, anchor, width) honouring the toggles."""
        cols = [
            ("file", "File", "w", 250),
            ("eps1", "eps1", "e", 60),
            ("eps2", "eps2", "e", 60),
            ("tp", "Throughput", "e", 100),
        ]
        if self.show_std_var.get():
            cols.append(("tp_s", "Throughput std", "e", 110))
        cols.append(("dm", "Mean delay", "e", 100))
        if self.show_std_var.get():
            cols.append(("dm_s", "Mean delay std", "e", 110))
        cols.append(("dx", "Max delay", "e", 100))
        if self.show_std_var.get():
            cols.append(("dx_s", "Max delay std", "e", 110))
        if self.show_good_var.get():
            cols += [
                ("good", "Goodput", "e", 100),
                ("dec", "Decoded", "e", 90),
                ("t", "t_mean", "e", 100),
            ]
        cols.append(("n", "n", "e", 55))
        return cols

    def _row_values(self, label, e1, e2, rec, columns):
        vmap = {
            "file": label,
            "eps1": f"{e1:.2f}",
            "eps2": f"{e2:.2f}",
            "tp": fmt_tp(rec["tp_mean"]),
            "tp_s": fmt_tp(rec["tp_std"]),
            "dm": fmt_delay(rec["dm_mean"]),
            "dm_s": fmt_delay(rec["dm_std"]),
            "dx": fmt_delay(rec["dx_mean"]),
            "dx_s": fmt_delay(rec["dx_std"]),
            "good": fmt_good(rec["good_mean"]),
            "dec": fmt_int(rec["dec_mean"]),
            "t": fmt_int(rec["t_mean"]),
            "n": str(rec["n"]),
        }
        return [vmap[c[0]] for c in columns]

    def _make_tree(self, parent, columns):
        frame = ttk.Frame(parent)
        ids = [c[0] for c in columns]
        tree = ttk.Treeview(frame, columns=ids, show="headings", selectmode="browse")
        for colid, heading, anchor, width in columns:
            tree.heading(colid, text=heading, command=lambda c=colid, t=tree: self._sort_tree(t, c, False))
            tree.column(colid, anchor=anchor, width=width, stretch=(colid == "file"))
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return frame, tree

    def _sort_tree(self, tree, col, reverse):
        rows = [(tree.set(iid, col), iid) for iid in tree.get_children("")]

        def keyf(pair):
            v = str(pair[0])
            try:
                return (0, float(v.replace(",", "")))
            except ValueError:
                return (1, v.lower())

        rows.sort(key=keyf, reverse=reverse)
        for idx, (_, iid) in enumerate(rows):
            tree.move(iid, "", idx)
        tree.heading(col, command=lambda: self._sort_tree(tree, col, not reverse))

    # ---- per-k graph ------------------------------------------------------
    @staticmethod
    def _short_label(path):
        b = os.path.basename(path)
        return b if len(b) <= 30 else b[:14] + "..." + b[-14:]

    def _diag_series(self, k, loaded_paths):
        """Diagonal (e1 == e2) sweep for one k.

        Returns (xs, {path: {"tp": [...], "dm": [...], "dx": [...]}}) with each
        y-list aligned to xs (NaN where a file lacks that eps point). Independent
        of the eps text box: always uses every eps1 == eps2 present for this k.
        """
        diag = sorted({e1 for (p, kk, e1, e2) in self.records
                       if kk == k and abs(e1 - e2) < 1e-9})
        series = {}
        for p in loaded_paths:
            ys = {"tp": [], "dm": [], "dx": []}
            has_any = False
            for e in diag:
                rec = self.records.get((p, k, e, e))
                if rec is None:
                    ys["tp"].append(float("nan"))
                    ys["dm"].append(float("nan"))
                    ys["dx"].append(float("nan"))
                else:
                    has_any = True
                    ys["tp"].append(rec["tp_mean"])
                    ys["dm"].append(rec["dm_mean"])
                    ys["dx"].append(rec["dx_mean"])
            if has_any:
                series[p] = ys
        return diag, series

    def _make_graph(self, parent, k, loaded_paths):
        """Frame holding a 3-subplot figure (throughput / mean delay / max delay)
        vs eps (e1 = e2), one line per pickle, all overlaid on shared axes."""
        frame = ttk.Frame(parent)
        diag, series = self._diag_series(k, loaded_paths)
        if not _HAVE_MPL:
            ttk.Label(frame, text="matplotlib not available - graphs disabled").pack(pady=8)
            return frame
        if not diag or not series:
            ttk.Label(frame, text="No eps1 = eps2 diagonal points for this k.").pack(pady=8)
            return frame

        fig = Figure(figsize=(9.5, 3.3), dpi=100)
        metrics = [("tp", "Throughput"), ("dm", "Mean delay (slots)"), ("dx", "Max delay (slots)")]
        for idx, (mkey, mlabel) in enumerate(metrics):
            ax = fig.add_subplot(1, 3, idx + 1)
            for p, ys in series.items():
                ax.plot(diag, ys[mkey], marker="o", markersize=3, linewidth=1.2,
                        label=self._short_label(p))
            ax.set_title(mlabel, fontsize=9)
            ax.set_xlabel("eps (e1 = e2)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.minorticks_on()
            ax.grid(True, which="major", alpha=0.35)
            ax.grid(True, which="minor", alpha=0.15, linestyle=":", linewidth=0.6)
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=max(1, min(len(labels), 3)), fontsize=7, frameon=False)
        fig.suptitle(f"k = {k}   (eps1 = eps2 sweep)", fontsize=10)
        fig.tight_layout(rect=(0, 0.1, 1, 0.94))

        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        canvas.draw()
        return frame

    # ---- build ------------------------------------------------------------
    def _ensure_records(self):
        """(Re)aggregate only when the selected file set changed; reuse the cache
        otherwise so column/option toggles re-render instantly."""
        key = tuple(self.files)
        if key != self._agg_files_key:
            self.records, self.all_k, self._load_errors = aggregate_files(self.files)
            self._agg_files_key = key
            for p, msg in self._load_errors.items():
                messagebox.showerror("Load error", f"{os.path.basename(p)}:\n{msg}")

    def _on_option_toggle(self):
        """Column-option checkbox handler: re-render immediately if files are loaded."""
        if self.files:
            self.build_tables()

    def build_tables(self):
        if not self.files:
            self._status("No files selected. Click 'Add...' to choose .pkl files.")
            return

        self._ensure_records()
        errors = self._load_errors

        if self.all_eps_var.get():
            req_eps = None
        else:
            req_eps = parse_eps(self.eps_entry.get())
            if not req_eps:
                self._status("No valid (eps1,eps2) tuples. Example: (0.1,0.1),(0.4,0.4)")
                return

        present_pk = {(p, k) for (p, k, _e1, _e2) in self.records}
        eps_by_pk = defaultdict(set)
        for (p, k, e1, e2) in self.records:
            eps_by_pk[(p, k)].add((e1, e2))

        columns = self._columns()
        try:
            prev_index = self.notebook.index(self.notebook.select())
        except Exception:
            prev_index = 0
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self._trees = {}
        self._built = []

        ks = sorted(self.all_k, key=_k_sort_key)
        loaded_paths = [p for p in self.files if p not in errors]
        total = 0
        skipped = 0
        for k in ks:
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=f"k = {k}")
            paned = ttk.Panedwindow(tab, orient="vertical")
            paned.pack(fill="both", expand=True)
            graph_frame = self._make_graph(paned, k, loaded_paths)
            tree_frame, tree = self._make_tree(paned, columns)
            paned.add(graph_frame, weight=3)
            paned.add(tree_frame, weight=4)
            self._trees[k] = tree
            for path in loaded_paths:
                if (path, k) not in present_pk:
                    continue  # this file simply has no such k
                label = os.path.basename(path)
                eps_list = sorted(eps_by_pk[(path, k)]) if req_eps is None else req_eps
                for (e1, e2) in eps_list:
                    rec = self.records.get((path, k, e1, e2))
                    if rec is None:
                        skipped += 1
                        continue
                    tree.insert("", "end", values=self._row_values(label, e1, e2, rec, columns))
                    self._built.append((k, path, e1, e2))
                    total += 1

        tabs = self.notebook.tabs()
        if tabs:
            self.notebook.select(min(prev_index, len(tabs) - 1))

        msg = f"{len(loaded_paths)} file(s)  -  {len(ks)} k tab(s)  -  {total} rows"
        if skipped:
            msg += f"  -  {skipped} (file,k,eps) combos had no data"
        if errors:
            msg += f"  -  {len(errors)} file(s) failed to load"
        self._status(msg)

    # ---- export -----------------------------------------------------------
    def export_csv(self):
        if not self._built:
            self._status("Nothing to export - click 'Build tables' first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export combined CSV",
            initialdir=_REPO_ROOT,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        header = [
            "k", "file", "eps1", "eps2",
            "throughput", "throughput_std",
            "mean_delay", "mean_delay_std",
            "max_delay", "max_delay_std",
            "goodput", "decoded", "t_mean", "n",
        ]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for (k, p, e1, e2) in self._built:
                r = self.records[(p, k, e1, e2)]
                w.writerow([
                    k, os.path.basename(p), e1, e2,
                    r["tp_mean"], r["tp_std"],
                    r["dm_mean"], r["dm_std"],
                    r["dx_mean"], r["dx_std"],
                    r["good_mean"], r["dec_mean"], r["t_mean"], r["n"],
                ])
        self._status(f"Exported {len(self._built)} rows to {os.path.basename(path)}")

    def _status(self, text):
        self.status_var.set(text)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="GUI table viewer for jammed-grid result pickles.")
    ap.add_argument("files", nargs="*", help="result .pkl files to preload")
    ap.add_argument("--eps", default=None, help='eps tuples, e.g. "(0.1,0.1),(0.4,0.4),(0.8,0.8)"')
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    app = AnalyzerApp(initial_files=args.files, initial_eps=args.eps)
    app.mainloop()


if __name__ == "__main__":
    main()
