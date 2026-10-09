# MP MH AC-RLNC Simulation

## Overview

The goal of this project is to **empirically evaluate how AC-RLNC performs
against a random jammer — one that blocks `K` randomly-chosen channels every
`RTT/α` time slots — and to compare it against SR-ARQ** under the same
adversarial conditions.

To trust those jamming results, both protocols are first **validated against the
paper** on the (non-jammed) **multipath (MP)** and **multipath multi-hop
(MP-MH)** erasure channels, by reproducing its throughput and in-order-delay
results. The implemented adaptive-causal coding scheme comes from:

> A. Cohen, G. Thiran, V. Bar Bracha, and M. Médard, "Adaptive Causal Network  
> Coding With Feedback for Multipath Multi-Hop Communications," *IEEE*  
> *Transactions on Communications*, vol. 69, no. 2, pp. 766–785, Feb. 2021.  
> (Preprint: arXiv:1910.13290.)

The work spans two protocols — **AC-RLNC** (the paper's scheme) and **SR-ARQ**
(uncoded Selective-Repeat ARQ baseline) — exercised on three network families,
each with its own drivers under `scripts/`:

- **Jamming** (`jamming_simulation/`). The main experiment. AC-RLNC and SR-ARQ on
  a network with a random jammer. *(Documented below.)*
  * Run AC-RLNC:
  ```python
  python scripts/jam_mp_mh_simulation.py
  ```
  * Run SR-ARQ:
  ```python
  python scripts/sr_arq_jam_simulation.py
  ```
- **AC-RLNC** (`mp_mh_network/`): The paper's protocol, validated against the
  paper in the MP setting (Fig. 11) and the MP-MH setting (Fig. 19). **Documented
  in detail below.**
  * Run MP:
  ```python
  python scripts/mp_simulation.py 
  ```
  * Run MP-MH:
  ```python
  python scripts/mp_mh_simulation.py
  ```
- **SR-ARQ** (`sr_arq/`): The uncoded baseline (also used by the paper). **Documented
  in detail below.**
  * Run MP:
  ```python
  python scripts/sr_arq_simulation.py
  ```
  * Run MP-MH:
  ```python
  python scripts/sr_arq_mpmh_simulation.py
  ```

## Repository Layout

```
mp_mh_network/        AC-RLNC protocol (MP + MP-MH). Also the shared
                      Path / Channel / Packet / feedback primitives reused
                      by the other network families.
sr_arq/               Uncoded Selective-Repeat ARQ baseline.
jamming_simulation/   Jamming network + jammer (K blocked channels / block).
scripts/              All simulation drivers, helpers, and plotting tools.
```

## Requirements

- Python 3.10+ (the code uses `match`/`case` and `X | None` type unions).
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  which pins `numpy`, `matplotlib`, and `networkx` (NetworkX provides the
  min-cut / max-flow capacity reference). The drivers add `mp_mh_network/` to
  `sys.path` automatically, so there is no install/packaging step — just run
  them from the repo root.
- Optional: the MP-MH driver's docstring references a conda env named
`mp_mh_ac_rlnc` (NetworkX from conda-forge); any environment with the above
packages works.

## Building Blocks

Every simulation is assembled from a small set of shared classes in
`mp_mh_network/` (plus `general/`). Each network family (AC-RLNC, SR-ARQ, jamming)
then subclasses these; the protocol-specific pieces are listed in that family's own
**Building Blocks**.

- **Slot clock:** Every active element (sender, node, receiver, jammer) exposes
  `run_step(t)` and is advanced once per time slot in a fixed order (each family's
  **Time-Slot Flow** shows the order). `general/GeneralUnit.py`'s `GeneralUnit` is
  the minimal base (`run_step` + logging) used by the `Jammer`; senders and
  receivers have their own `General*` bases below.
- **`Packet` (`Packet.py`):** Base class for each type of packet sent on the network. For example:
  `FeedbackPacket` (`ACK` / `NACK`), and protocol-specific payloads: `RLNCPacket` (AC-RLNC) or `DataPacket` (SR-ARQ).
- **`PacketID` (`PacketID`):** Identifier for packets.
- **`Channel` (`Channels.py`):** The pipe that transfers packets bewteen 2 units.
  Has a fixed propagation delay.
- **`ForwardChannel` (`Channels.py`):** A BEC `Channel` that erases each packet with probability `ε`.
- **`Path` (`Channels.py`):** bundles one link's two directions: a forward `ForwardChannel` and a reliable
  feedback `Channel` at the same propagation delay, tagged with a global path index.
- **Per-role path wrappers:** A unit never drives a bare `Path`. The sender side
  wraps it as **`GeneralSenderPath`** (adds the feedback history and the online rate
  estimate `r = 1 − ε̂`); the receiver side as **`ReceiverPath`** (tracks arrivals).
  Each protocol subclasses these further (`SimSenderPath`, `SRSenderPath`, `JamPath`
  below).
- **Unit bases (`Sender.py` / `Receiver.py`):** `GeneralSender` and `GeneralReceiver`
  hold the per-path feedback plumbing and round-robin path polling; the concrete
  `SimSender` / `NodeSender` / `SRSender` … and `SimReceiver` / `NodeReceiver` /
  `SRSimReceiver` … specialize them.
- **Assembly + clock (`Network.py`):** `Network` (single-hop) and `MhNetwork`
  (multi-hop, which splits the end-to-end RTT across the hops) wire the units
  together, run the per-slot loop, and collect `SimulationStats` (throughput +
  in-order delay). Every concrete network subclasses one of these.

## AC-RLNC Network (`mp_mh_network/`)

A discrete-time, slotted simulation of the AC-RLNC protocol. Forward links are
independent **binary erasure channels (BEC)**: in each slot a transmitted packet
is erased with probability `ε`. Feedback (ACK/NACK) is **reliable** (never
erased) and delayed by the same propagation as the forward link, so the sender
reacts to channel state one RTT late.

### Time-Slot Flow

Each slot advances the whole network in a fixed order: the **sender** first, then
the **intermediate nodes in hop order**, then the **receiver**. A packet sent this
slot reaches the next unit after that link's propagation delay, and its feedback
returns the same way one RTT later.

```text
Order within one time slot (MP-MH, H = 3):

  SimSender ──▶ Node₁ ──▶ Node₂ ──▶ SimReceiver
    (1st)       (2nd)     (3rd)       (last)

(MP, H = 1: SimSender ──▶ SimReceiver, no nodes.)
```

### Building Blocks

On top of the shared **Building Blocks** above, this network adds (`mp_mh_network/`):

- **`RLNCPacket` (`Packet.py`):** The forward payload. Carries the set of
  information-packet indices it mixes plus a type: `NEW` / `FEC` / `FB_FEC` at the
  source, or `CORRECTION` / `DROPPED` at a node.
- **`CodedEquation` (`CodedEquation.py`):** One coded packet viewed as an equation
  over its unknown information packets. The receiver decodes a window once
  #independent-equations ≥ #unknowns (rank tracking: no finite-field arithmetic).
- **`FeedbackSource` (`feedback_source.py`):** The `HBH` / `E2E` enum selecting where
  the sender's feedback comes from (see **Feedback (ACK/NACK) Types**).
- **`SimSender` + `SimSenderPath` (`Sender.py`):** The source. It drives each path as
  a `SimSenderPath`, a `GeneralSenderPath` that adds the per-path a-priori-FEC budget
  `mp`.
- **`Node` = `NodeReceiver` + `NodeSender` (`Node.py`, `Receiver.py`, `Sender.py`):**
  An intermediate relay that recodes and forwards (`NodeSender` drives its output
  paths as `GeneralSenderPath`s).
- **`SimReceiver` (`Receiver.py`):** The destination decoder: in-order delivery plus
  per-slot feedback.
- **`MPNetwork` / `MpMhNetwork` (`Network.py`):** The single-hop and multi-hop
  assemblies.

### MP Network Architecture (`mp_simulation.py` → `MPNetwork`)

`MPNetwork` is the single-hop (`H = 1`) multipath setting: one sender and one
receiver joined by `P = 4` parallel BEC paths, with no intermediate nodes. The
single hop carries the full end-to-end one-way delay `RTT/2`, and every path has
its own reliable feedback channel back to the sender. It is a single joint
AC-RLNC stream striped across the four paths.

The architecture matches the paper (Fig. 2–3 top, Fig. 11):

```text
MPNetwork — H = 1 hop, P = 4 paths, one-way delay = RTT/2  (RTT = 20 slots)

                     forward: coded RLNC packets  ───────────▶
               ┌─── path 1  (BEC, ε₁) ────┐
   SimSender ──┼─── path 2  (BEC, ε₂) ────┼──▶ SimReceiver
               ├─── path 3  (ε₃ = 0.2) ───┤    (decodes in order)
               └─── path 4  (ε₄ = 0.8) ───┘
                     ◀───────────  feedback: reliable ACK/NACK, one per path
```

### MP-MH Network Architecture (`mp_mh_simulation.py` → `MpMhNetwork`)

`MpMhNetwork` is the multi-hop setting: `H = 3` hops of `P = 4` parallel BEC
paths each, with `H − 1 = 2` intermediate `Node`s between the sender and the
receiver. The end-to-end one-way delay is split evenly across hops
(`hop_prop_delay = (RTT/2)/H`, `hop_rtt = RTT/H`).

Each hop's four paths carry the per-hop erasure rates `ε_{p,h}` from the paper's
4×3 matrix (rows = paths, columns = hops; built in
`scripts/mh_epsilon_matrix.py`), with `ε₁, ε₂` swept in `[0.1, 0.8]`:

```text
          Hop 1   Hop 2   Hop 3
 Path 1    ε₁      0.6     0.3
 Path 2    0.8     ε₁      ε₁
 Path 3    0.2     ε₂      0.7
 Path 4    ε₂      0.4     ε₂
```

The architecture matches the paper (Fig. 12, Fig. 19). Like the sender and
receiver, each intermediate node terminates its four input paths and re-launches
on four output paths — the stream fans out across `P` paths, is re-collected at
the next node, fans out again, and so on:

```text
MpMhNetwork — H = 3 hops, P = 4 BEC paths per hop, per-hop one-way delay = (RTT/2)/H  (RTT = 12)

               ┌── path 1 ──┐           ┌── path 1 ──┐           ┌── path 1 ──┐
   SimSender ──┼── path 2 ──┼── Node₁ ──┼── path 2 ──┼── Node₂ ──┼── path 2 ──┼──▶ SimReceiver
               ├── path 3 ──┤           ├── path 3 ──┤           ├── path 3 ──┤
               └── path 4 ──┘           └── path 4 ──┘           └── path 4 ──┘
```

Every path also carries reliable ACK/NACK feedback back one hop; how the
sender's feedback is obtained (hop-by-hop vs end-to-end) is covered in
**Feedback (ACK/NACK) Types**. The per-node recoding and the natural matching
that assigns physical paths to global paths are described in **Sender Operation**
and **Intermediate Node Operation** below.

### Sender Operation

The sender (`SimSender`) implements the paper's MP AC-RLNC packet-scheduling
algorithm (Algorithm 1). Each slot, after folding in the feedback that just
arrived, it allocates the `P` paths among three kinds of transmission:

- **New** coded packets: Coded packet with new information in them.
- **A-priori FEC:** After `k = P(RTT − 1)` new packets, `m_p = round(ε̂_p(RTT − 1))`
planned repetitions per path to pre-empt the expected erasures.
- **Feedback FEC (FB-FEC):** Extra repetitions triggered *a posteriori* when the
degrees-of-freedom gap `Δ = P(d − 1 − th)` is positive, where `d = mdg/adg` is
the missing-to-added DoF ratio estimated from feedback.

The split between new packets and FB-FEC across paths is decided by
**bit-filling** (discrete water-filling, paper Prop. 1).  
A sliding window bounds the number of distinct undecoded information  
packets in flight; once it reaches the size limit `ō`, the sender
only repeats until the receiver has decoded the window.

**Natural Matching (decentralized balancing, Algorithm 2).** In the MP-MH network,  
every slot the sender ranks its paths by estimated rate `r = 1 − ε̂` and
publishes the ordering; each node then re-matches its local paths to global-path
labels in descending-`r` order, so the strongest local path at every hop is
stitched into the strongest global path. This minimizes the bottleneck effect
caused by hop-to-hop rate variation.

**Sender parameters.** The AC-RLNC knobs inside `SimSender` are its two tunable
parameters (`th`, `ō`) and the generation size `k`, plus the quantities it tracks
online from feedback:


| Quantity (code)              | Paper symbol | Definition / value                                     | Notes                                                                        |
| ---------------------------- | ------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `number of packets in window`| `k`          | `k = P·(RTT − 1)`                                          | end-window: # new coded packets per generation before the a-priori FEC burst |
| `max_allowed_overlap` | `ō` | `ō = 2k` | window size limit on the number of distinct undecoded packets in flight (`= 152` for MP, `88` for MP-MH); paper uses `ō = 2k` **(paper)** |
| `threshold` | `th` | `th = 0` | tunable retransmission threshold (appears in `Δ`); paper uses `th = 0` **(paper)** |
| `path.mp`                    | `m_p`        | `round(ε̂_p·(RTT − 1))`                                | per-path a-priori FEC count **(paper)**                                      |
| `path.r`, `path.epsilon_est` | `r_p`, `ε_p` | `r = 1 − ε̂`; `ε̂ = NACKs / total feedback`            | per-path rate / erasure estimate from feedback                               |
| `d`                          | `d`          | `mdg / adg`                                            | DoF rate (missing ÷ added DoF)                                               |
| `delta`                      | `Δ`          | `P·(d − 1 − th)`                                       | FB-FEC triggered when `Δ > 0`                                                |
| `md1, md2, ad1, ad2`         | —            | paper eqs. (2)–(3)                                     | with / without-feedback DoF counts                                           |
| bit-filling                  | Prop. 1      | lowest-`r` paths cover `Δ`; the rest carry new packets | discrete water-filling allocation                                            |


### Intermediate Node Operation

Each `Node` (`NodeReceiver` + `NodeSender`) sits between two hops and, like the
sender and receiver, terminates its `P` input paths and re-launches on its `P`
output paths:

- **Receive (`NodeReceiver`).** Collects arrivals on its `P` input paths, buffers
the information indices it has seen (keeping new-packet indices and
correction-packet indices in separate buffers), and emits one per-hop ACK/NACK
per path back up the previous hop.
- **Selective mixing (`NodeSender`).** Recodes and forwards on its `P` output
paths: it sends `NEW`-type packets built from its **new** buffer on the paths
currently carrying new data, and `CORRECTION` packets built from its
**FEC/FB-FEC** buffer on the rest. Keeping these two buffers separate — rather
than mixing everything together — is the paper's **selective mixing** (§V-A): it
keeps FEC/FB-FEC repetitions useful instead of diluting them with new data. If
an input slot was erased this slot, the node still fills the outgoing slot from
its buffer (re-coding) rather than propagating the gap, which keeps each global
path at its bottleneck-hop (min-cut) rate.

### Receiver Operation

The receiver (`SimReceiver`) terminates the last hop's `P` paths:

- **Decode.** Each arrival adds its information indices as a new coded equation;
the current window is decoded as soon as the number of independent equations
reaches the number of unknown information packets (the simulator tracks ranks,
not actual coded coefficients). Newly decoded packets are released **in order**
and time-stamped for the delay metric.
- **Feedback.** Every slot it sends one per-hop ACK/NACK per path back to the last
node (always on). Under **E2E** feedback it additionally sends one end-to-end
ACK/NACK per global path straight back to the sender over dedicated feedback
channels (see **Feedback (ACK/NACK) Types** below).

### Feedback (ACK/NACK) Types

Feedback is reliable and delayed by the link propagation. Which feedback the
**sender** consumes is selected by `FeedbackSource`
(`mp_mh_network/feedback_source.py`), passed to `MpMhNetwork`. In `MPNetwork`
there is a single hop, so HBH and E2E coincide.

- **Per-hop feedback (always on):** Every receiver (each `NodeReceiver` and the
`SimReceiver`) emits exactly one ACK or NACK per path per slot on that path's
feedback channel: **ACK** if a packet arrived on that path this slot, **NACK**
(tagged with the slot the missing packet should have occupied) if nothing
arrived. This drives each path's online `ε̂` / `r` estimate.
- **HBH (hop-by-hop)** (`FeedbackSource.HBH`, paper Fig. 19 bottom): The sender
and every node act on the feedback from the **next hop only**: the `SimSender`
reacts to delivery at `Node₁`, `Node₁` to delivery at `Node₂`, and the last node
to delivery at the `SimReceiver`. Together with node recoding, each hop is
corrected locally.
- **E2E (end-to-end)** (`FeedbackSource.E2E`, paper Fig. 19 top): The sender's
feedback comes **directly from the `SimReceiver`** over dedicated per-global-path
end-to-end feedback channels carrying the full end-to-end delay (`RTT/2`). Each
slot the receiver emits exactly one feedback per global path: **ACK** for every
global label that arrived, **NACK** for every label erased on the last hop; the
sender routes each into the matching path for its `ε̂` / `r` estimate.
Intermediate nodes still recode and still exchange per-hop feedback among
themselves; only the **sender's** view is end-to-end.

> **Known bug (HBH), flagged for future work.** In HBH mode each intermediate
> node currently generates and sends **its own** feedback (ACK/NACK about what
> *it* received on its input hop) one hop upstream, instead of **relaying the
> feedback coming from the hop after it** back toward the sender (which would
> propagate downstream delivery status upstream, as E2E feedback does). As a
> result, an upstream sender only ever sees the delivery status of its immediate
> next hop, never the relayed downstream / end-to-end status. This should be
> reconciled with the paper's hop-by-hop semantics.

### Metrics

Collected per run in `Network.collect_stats` and aggregated (mean ± std over the
150 realizations) by the drivers:

- **Normalized throughput** `η = (# information packets decoded in order at the receiver) / (# time slots)`.
- **In-order delivery delay** `D`, per information packet
`= decode_time − first_transmission_time`; reported as **mean** and **max**. The
ACK propagation time is **not** counted (paper Def. 2).
- **Reference capacity surface** (red, in the plots): the min-cut / max-flow of
the layered BEC `= minₕ Σ_p (1 − ε_{p,h})` (MP: `Σ_p (1 − ε_p)`), from
`scripts/mh_min_cut_capacity.py` (NetworkX, cross-checked against the closed
form). Achieved throughput should stay at or below this surface.

(The paper additionally compares delay to a genie-aided lower bound
`RTT/2 + 1/(1 − ε̄)`; the drivers plot the capacity surface as the throughput
reference.)

### Simulation Parameters

Values below are what the two drivers use (constructor arguments of `MPNetwork` /
`MpMhNetwork`, plus the sweep's Monte-Carlo count). Those that reproduce the paper
are marked **(paper)**; the rest are implementation choices, tagged as such (you
may want to confirm or adjust them). The AC-RLNC tunable knobs `th` and `ō`, and
the generation size `k = P·(RTT − 1)`, are listed under **Sender operation**.


| Parameter (code)                   | Paper symbol      | MP (`mp_simulation.py`) | MP-MH (`mp_mh_simulation.py`)             | Notes                                                         |
| ---------------------------------- | ----------------- | ----------------------- | ----------------------------------------- | ------------------------------------------------------------- |
| `num_paths`                        | `P`               | 4                       | 4                                         | **(paper)**                                                   |
| `num_hops`                         | `H`               | 1                       | 3                                         | **(paper)**                                                   |
| `prop_delay` / `global_prop_delay` | `t_prop` (slots)  | 10 → `RTT = 20`         | 6 → `RTT = 12`                            | **(paper)** RTT; `RTT = 2·t_prop`                             |
| per-hop one-way delay (derived)    | `t_prop,h`        | 10                      | 2                                         | `t_prop / H`                                                  |
| per-hop RTT (derived)              | —                 | 20                      | 4                                         | `RTT / H`                                                     |
| `path_epsilons`                    | `ε_p` / `ε_{p,h}` | `[ε₁, ε₂, 0.2, 0.8]`    | paper 4×3 matrix (see MP-MH architecture) | **(paper)**; `ε₁, ε₂ ∈ [0.1, 0.8]`                            |
| `num_packets_to_send`              | —                 | 200                     | 500                                       | *implementation choice; paper states no packet count*         |
| `max_iterations` (slot cap)        | —                 | `None`                  | 40000                                     | *implementation safeguard*                                    |
| `initial_epsilon`                  | —                 | 0.5                     | 0.5                                       | *implementation choice (initial per-path ε̂ before feedback)* |
| `feedback_source`                  | —                 | n/a (`H = 1`)           | `HBH` or `E2E`                            | **(paper)** Fig. 19: top = E2E, bottom = HBH                  |
| `NUM_ITERATIONS` (sweep)           | —                 | 150                     | 150                                       | **(paper)** "averaged on 150 realizations"                    |


### Running The Simulations

Both drivers run from the repository root; each sweeps the two free erasure rates
`(ε₁, ε₂)` over a grid, runs many Monte-Carlo channel realizations per grid
point, saves the aggregated results to a `.pkl`, and renders a 3-panel 3-D figure
(normalized throughput, mean in-order delay, max in-order delay vs `ε₁, ε₂`) with
the min-cut capacity surface overlaid.

**MP (single hop, `H = 1`), paper Fig. 11 setting:**

```bash
python scripts/mp_simulation.py
```

**MP-MH (`H = 3`), paper Fig. 19 setting:**

```bash
python scripts/mp_mh_simulation.py
```

Key knobs (edit near the top of `__main__` / `_run_main`):

- `mp_simulation.py`: `NUM_PATHS`, `PROP_DELAY` (→ `RTT = 2·PROP_DELAY`),
`THRESHOLD`, `O_BAR`, `EPS3`, `EPS4`, `NUM_PACKETS_TO_SEND`, `NUM_ITERATIONS`,
`LOAD_EXISTING`, `RESULTS_FILE`.
- `mp_mh_simulation.py`: `RTT`, `NUM_HOPS`, `NUM_PATHS`, `THRESHOLD`, `O_BAR`,
`NUM_PACKETS_TO_SEND`, `MAX_ITERATIONS`, `NUM_ITERATIONS`, `FEEDBACK_SOURCE`
(`HBH`/`E2E`), `PARALLEL_WORKERS`, `LOAD_EXISTING`. The ε matrix comes from
`scripts/mh_epsilon_matrix.py`.

(`mp_mh_simulation.py` runs the sweep across processes by default; set
`PARALLEL_WORKERS = 1`, or `DEBUG = True`, for a single-process run with verbose
per-slot logging.)

### Output Files & Re-Plotting

- `mp_simulation.py` → `mp_simulation_results.pkl` and `mp_performance_3d.png`.
- `mp_mh_simulation.py` → `mp_mh_simulation_results_global_RTT_<RTT>_<HBH|E2E>.pkl`
and the matching `..._3d_...png` (the feedback tag keeps HBH and E2E outputs
from colliding).

To re-render without re-simulating:

- set `LOAD_EXISTING = True` in the driver (reads its `.pkl` and re-plots), or
- `python scripts/plot_saved_results.py <results.pkl>`: Plots one pickle, or
several side-by-side for protocol comparison (`--paper` gives a stacked
paper-style figure), or
- `python scripts/analyze_results_gui.py`: A Tkinter table viewer; primarily
aimed at the jammed-grid sweeps but reads the same `SimulationStats` pickles.

---

## SR-ARQ Simulation (`sr_arq/`)

An uncoded **Selective-Repeat ARQ** simulation, the baseline AC-RLNC is measured
against (and the protocol the jamming experiment pits against AC-RLNC). It runs on
the **same** discrete-time BEC forward-channel + reliable per-slot feedback model
as the AC-RLNC network, reusing `GeneralSender` / `GeneralReceiver` / `Path` /
`Channel` / `SimulationStats`, so the two protocols are directly comparable on the
same channel realizations.

### Time-Slot Flow

The multi-hop `SRMpMhNetwork` ticks each slot in a fixed order: the **sender**, then
the **`SRNode`s in hop order**, then the **receiver**. (The single-hop `SRNetwork`
is just sender → receiver.)

```text
Order within one time slot (MP-MH, H = 3):

  SRSimSender ──▶ SRNode(hop 1) ──▶ SRNode(hop 2) ──▶ SRSimReceiver
     (1st)           (2nd)             (3rd)             (last)
```

### Building Blocks

On top of the shared **Building Blocks** above, this baseline adds (`sr_arq/`):

- **`DataPacket` + `SRType` (`SRPacket.py`):** The uncoded forward payload: one
  information packet = its sequence number `seq`.
- **`SRSenderPath` + `SRSender` / `SRSimSender` (`SRSender.py`):** `SRSenderPath` is a
  `GeneralSenderPath` that also records which `seq` went out each slot (to resolve
  slot-based NACKs). `SRSender` is the shared-multipath source; `SRSimSender` is the
  decoupled per-chain source.
- **`SRReceiver` / `SRSimReceiver` (`SRReceiver.py`):** `SRReceiver` is the single-hop
  global-in-order receiver; `SRSimReceiver` is the decoupled per-chain receiver (plus
  stats and end-to-end feedback emission).
- **`SRNode` = `SRNodeReceiver` + `SRNodeSender` (`SRNode.py`):** The hop-by-hop relay.
- **`SRFeedbackMode` (`sr_feedback.py`):** The feedback / relay modes (see **Feedback
  (ACK/NACK) Modes**).
- **`SRNetwork` / `SRMpMhNetwork` (`SRNetwork.py`):** The single-hop and multi-hop
  assemblies. **`SRJamMpMhNetwork`** (`SRJamNetwork.py`) is the jammed variant
  (covered in **Jamming Simulation**).

### MP Network Architecture (`sr_arq_simulation.py` → `SRNetwork`)

Single hop, `P = 4` paths, no relays, one `SRReceiver` reordering globally. The
`independent` flag picks the sender:

- `independent=False` → `SRSender` (**shared multipath**): one global `seq` stream
  striped across all paths, with retransmissions rerouted onto whichever path is
  free. A stronger-than-standard multipath design.
- `independent=True` → `SRSimSender` (**round-robin**): each path owns a static
  round-robin slice of the `seq` space (path `i` → seqs `i+1, i+1+P, …`), no
  rerouting.

Both still deliver globally in order at `SRReceiver`, so one stalled path holds up
global delivery. (The paper's canonical "SR-ARQ independently on each path"
baseline — where a bad path never blocks a good one — is assembled at the driver
level as `P` separate single-path `SRNetwork`s; see **Running**, proto
`sr_perpath`.)

```text
SRNetwork — H = 1 hop, P = 4 paths  (RTT = 20 slots)

               ┌─── path 1  (BEC, ε₁) ────┐
   SRSender ───┼─── path 2  (BEC, ε₂) ────┼──▶ SRReceiver
   (or         ├─── path 3  (ε₃ = 0.2) ───┤    (global in-order)
   SRSimSender)└─── path 4  (ε₄ = 0.8) ───┘
                    ◀─── reliable ACK/NACK, one per path
```

### MP-MH Network Architecture (`sr_arq_mpmh_simulation.py` → `SRMpMhNetwork`)

`H = 3` hops, `P = 4` **independent chains**: each chain is a single path per hop
with its own `SRNode` per hop, and chains never mix. `SRSimSender` owns the hop-0
paths (per-chain round-robin `seq` slices), there is one `SRNode` per (chain,
hop), and `SRSimReceiver` terminates every chain's last hop with **decoupled
per-chain in-order** delivery. The end-to-end one-way delay is split across the
hops (`hop_rtt = RTT/H`), as in `MpMhNetwork`; `H = 1` reduces to the single-hop
decoupled model.

```text
SRMpMhNetwork — P = 4 independent chains, H = 3 hops each  (end-to-end RTT = 12; no cross-chain mixing)

                     hop 1          hop 2          hop 3
                 ┌──────── SRNode ──────── SRNode ────────┐
 SRSimSender ────┼──────── SRNode ──────── SRNode ────────┼──▶ SRSimReceiver
                 ├──────── SRNode ──────── SRNode ────────┤      (decoupled
                 └──────── SRNode ──────── SRNode ────────┘       per-chain in-order)
```

Contrast with AC-RLNC MP-MH, where each hop fans across the `P` paths and the
`Node` re-matches / mixes them at runtime: here the chains are fixed and
independent by **Natural Matching**. This matches the paper's architecture.

### Sender Operation

`SRSender` / `SRSimSender`:

- **Lowest-seq-first; retransmits before new.** The retransmit queue only holds
  seqs lower than the next new seq, so draining it first is exactly
  lowest-seq-first.
- **NACK-driven retransmission** (feedback is lossless, so HBH needs no timer): a
  NACK is slot-based `(global_path_id, creation_time)`; the sender maps it back to
  a `seq` via a per-path `creation_time → seq` record. ACKs carry the `seq`
  directly, are processed before NACKs, and an already-ACKed seq is never
  re-queued.
- **Sliding window = flow control, not reliability.** A new seq is admitted only
  while fewer than `window` of that stream's packets are outstanding. In the
  decoupled per-chain model (`SRSimSender`, used by MP-MH and jamming) this cap is
  **per chain**; in the shared single-hop `SRSender` it is one global window. A
  repeatedly-erased low seq freezes the window and throttles throughput below the
  link rate (real SR-ARQ behaviour); `window = None` is unbounded (front-loads,
  approaches capacity).
- `SRSender` (shared) keeps one stream across all paths with rerouted retransmits;
  `SRSimSender` (decoupled) runs an independent per-path slice with a per-path
  window and no rerouting, plus a `packets_per_path` quota and `close_admission`
  (used by the fixed-horizon drain).

(The three end-to-end feedback modes change *how the sender learns about losses*;
see **Feedback (ACK/NACK) Modes**.)

### Intermediate Node Operation

Each `SRNode` = `SRNodeReceiver` + `SRNodeSender`:

- **Full per-hop SR-ARQ** (`HBH` and `E2E_FULL_ARQ`): The node ACK/NACKs upstream,
  buffers received seqs, forwards them downstream with its own SR-ARQ, and
  retransmits on downstream NACKs.
- **Forwarding discipline:** Out-of-order by default (forward the lowest
  not-yet-forwarded seq, skip gaps; the gap is filled later by the upstream link's
  own retransmission). `in_order_forwarding=True` makes each node a full per-hop
  SR-ARQ endpoint (release only the contiguous in-order prefix), adding per-hop
  head-of-line blocking and higher delay (the paper's "full SR-ARQ at each node").
- **`node_queue_size`:** Bounds the relay buffer. When the held
  (received-but-not-yet-next-hop-ACKed) backlog is full, a new arrival is refused
  (NACK) so the upstream keeps and later retransmits it (hop-by-hop backpressure).
  `None` = unbounded.
- **Best-effort forwarders** (`E2E_FORWARD_ONLY`, `E2E_TIMEOUT`): The node forwards
  each arrival once with no per-hop retransmit / feedback; `in_order_forwarding`
  and `node_queue_size` are inactive.

### Receiver Operation

- `SRReceiver` (single hop): A global reorder buffer delivers seqs in order;
  built-in per-slot ACK (on arrival) / NACK (on an empty slot) go back upstream.
- `SRSimReceiver` (multi-hop final receiver): A **separate in-order frontier per
  chain** (keyed by `global_path_id`), so a stalled bad chain never blocks a good
  one. It records per-seq delivery times and per-chain finish time / count for the
  sum-of-per-chain-rates throughput, and in the E2E modes emits one end-to-end
  feedback per chain back to the sender (slot-based, seq-based, or ACK-only; see
  below).

### Feedback (ACK/NACK) Modes

Selected by `SRFeedbackMode` (`sr_arq/sr_feedback.py`), passed to `SRMpMhNetwork`
(at `H = 1` all modes coincide). `HBH` reads feedback on each hop's per-path
channel; the three E2E modes build one dedicated per-chain **end-to-end** channel
(full end-to-end delay) that carries feedback straight from the receiver to the
sender — so the sender's window must then be sized on the end-to-end RTT, not the
per-hop RTT.

- **`HBH`** (default; paper Fig. 19 **bottom**): The sender hears the first node,
  and every `SRNode` runs full per-hop SR-ARQ. Per-path rate = min-cut (bottleneck
  hop).
- **`E2E_FORWARD_ONLY`** (paper Fig. 19 **top**): Best-effort relays; the receiver
  sends **slot-based** end-to-end feedback (one per chain per slot: ACK the
  delivered seq, else NACK whose `creation_time = t − e2e_delay` resolves to a
  seq). A packet survives only if it clears all `H` hops, so per-path rate = product
  of the per-hop rates.
- **`E2E_FULL_ARQ`** (mirrors the AC-RLNC E2E structure): Full per-hop ARQ relays,
  but the sender hears only the receiver; **seq-based** selective-repeat feedback
  (ACK received seqs, NACK genuine per-chain gaps), with retransmission
  rate-limited to once per end-to-end RTT plus a tail backstop. Per-path rate sits
  between the other two.
- **`E2E_TIMEOUT`** (the external uncoded SR-ARQ baseline): Best-effort relays,
  **ACK-only** end-to-end feedback; the sender recovers losses purely with a
  retransmission **timer** (resend an unacked seq once it is one end-to-end RTT
  past its last send). Per-path rate = product of the per-hop rates.

### Metrics

Same pipeline and definitions as AC-RLNC:

- **Normalized throughput:** In-order-delivered packets per slot. The decoupled
  models (per-path, multi-hop) use the **sum of per-chain rates**
  (`delivered_c / finish_time_c`), so a slow chain does not drag the others down.
  (Fixed-horizon runs instead snapshot total in-order deliveries ÷ horizon.)
- **In-order delivery delay** (mean, max): `delivery_time − first_transmission_time`
  (ACK propagation not counted). `D_max` is the single worst packet over all paths.

### Simulation Parameters

| Parameter (code) | MP (`sr_arq_simulation.py`) | MP-MH (`sr_arq_mpmh_simulation.py`) | Notes |
| --- | --- | --- | --- |
| `num_paths` (`P`) | 4 | 4 | **(paper)** |
| `num_hops` (`H`) | 1 | 3 | **(paper)** |
| `RTT` (slots) | 20 | 12 | **(paper)**; MP-MH per-hop RTT = `RTT/H` = 4 |
| `path_epsilons` | `[ε₁, ε₂, 0.2, 0.8]` | paper 4×3 matrix | **(paper)**; `ε₁, ε₂ ∈ [0.1, 0.8]` |
| `window` (`SR_WINDOW`) | `RTT − 1 = 19` | `2·(hop_rtt − 1) = 6` | per-chain flow control (one global window in the shared single-hop `SRSender`); E2E modes want an end-to-end-RTT window, e.g. `2·(RTT−1)` |
| `independent` | shared / round-robin | — | MP only: `SRSender` vs `SRSimSender` |
| `feedback_mode` | — (`H = 1`) | `E2E_FORWARD_ONLY` / `E2E_FULL_ARQ` / `E2E_TIMEOUT` | **(paper)** Fig. 19: `HBH` = bottom, `E2E_FORWARD_ONLY` = top |
| `in_order_forwarding` | — | `False`| per-hop in-order relay (HOL delay) |
| `node_queue_size` | — | `None` | relay backpressure buffer; `None` = unbounded |
| `packets_per_path` | — | `None` | equal per-chain quota; `None` = unlimited |
| `num_packets_to_send` | 500 | 500 | implementation; MP-MH uses a fixed horizon when `max_iterations` is set |
| `max_iterations` | 20000 (cap) | 150 (measurement horizon) | implementation |
| `NUM_ITERATIONS` | 150 | 150 | **(paper)** "averaged on 150 realizations" |

### Running

**MP (single hop, `H = 1`), paper Fig. 11 reference** (SR-ARQ should sit clearly
below AC-RLNC):

```bash
python scripts/sr_arq_simulation.py
```

Pick the protocol(s) via the `protos` list in `sc_run_main`: `"sr"` (shared),
`"sr_indep"` (round-robin, global in-order), `"sr_perpath"` (per-path independent
= the paper's baseline), `"sr_mpmh"` (multi-hop decoupled), `"ac"` (MP AC-RLNC
overlay for side-by-side). Knobs: `NUM_HOPS`, `SR_WINDOW`, `NUM_PACKETS_TO_SEND`,
`NUM_ITERATIONS`, `PARALLEL_WORKERS`.

**MP-MH (`H = 3`), paper Fig. 19:**

```bash
python scripts/sr_arq_mpmh_simulation.py
```

Runs two settings: `best` (one global path from the best path of each hop) and
`matched` (`P` natural-matched chains). Knobs: `SR_FEEDBACK_MODE` (`HBH` → Fig. 19
bottom; `E2E_FORWARD_ONLY` → Fig. 19 top; `E2E_FULL_ARQ` / `E2E_TIMEOUT` extra),
`SR_WINDOW`, `IN_ORDER_FORWARDING`, `NODE_QUEUE_SIZE`, `PACKETS_PER_PATH`,
`MAX_ITERATIONS`, `NUM_ITERATIONS`. Set `LOAD_EXISTING = True` to replot from the
pickle. Runs are heavy (grid × 150 iters × 4 chains); reduce iterations for a quick
check.

### Output Files & Re-Plotting

- `sr_arq_simulation.py` → `sr_arq_mp_results.pkl` + `sr_arq_mp_compare.png`.
- `sr_arq_mpmh_simulation.py` → `sr_arq_mpmh_results_<MODE>_…_window_<w>_RTT_<rtt>_in_order_forwarding_<bool>_node_queue_size_<n>.pkl`
  (+ matching `…_compare_….png`); the filename is tagged with the feedback mode and
  knobs so `HBH` / `E2E` runs never collide.
- Re-plot with `LOAD_EXISTING = True`, or `python scripts/plot_saved_results.py <pkl> […]`
  (multi-series overlay; `--paper` for the stacked figure), or
  `python scripts/analyze_results_gui.py` (sortable table GUI).

---

## Jamming Simulation (`jamming_simulation/`)

The project's **main experiment**: run AC-RLNC and SR-ARQ on the *same* jammed
multi-path multi-hop network and compare how each holds up as a jammer blocks
more of the channel. The jammed network reuses both protocol stacks unchanged,
only the forward links become jammable and a `Jammer` is added.

**Feedback: end-to-end only.** Because of the HBH feedback bug (see the AC-RLNC
**Feedback (ACK/NACK) Types** section), every jamming run uses end-to-end feedback:
**AC-RLNC** with `FeedbackSource.E2E`, **SR-ARQ** with
`SRFeedbackMode.E2E_FORWARD_ONLY`.

### Time-Slot Flow

Each slot advances in a fixed order: the **jammer** first (at the start of every
jamming block it re-draws the blocked set; otherwise it keeps the current one),
then the **sender**, then the **nodes in hop order**, then the **receiver**.

```text
Order within one time slot:

  Jammer ──▶ sender ──▶ Node(hop 1) ──▶ Node(hop 2) ──▶ receiver
   (1st)      (2nd)        (3rd)           (4th)          (last)
```

### Building Blocks

On top of the shared **Building Blocks** above (and each protocol's own), the jammer
adds (`jamming_simulation/`):

- **`Jammer` (`Jammer.py`):** A `GeneralUnit` that each jamming block picks `k`
  forward links at random and blocks them.
- **`JamPath` + `JamForwardChannel` (`JamChannels.py`):** `JamPath` is a `Path` whose
  `JamForwardChannel` (a `ForwardChannel`) drops every packet while the jammer has it
  blocked; `jam_path()` / `unjam_path()` toggle it.
- **`JamMpMhNetwork` (`JamNetwork.py`):** AC-RLNC under jamming.
- **`SRJamMpMhNetwork` (`sr_arq/SRJamNetwork.py`):** SR-ARQ under jamming (subclasses
  `SRMpMhNetwork`).

### The Jammer

Each **jamming block** the `Jammer` picks `k` forward links **uniformly at random**
out of all `P·H` and blocks them for the whole block; a blocked link's forward
channel drops every packet that slot (a forced erasure on top of its BEC loss). A
block lasts `RTT/α` slots, after which a new random set of `k` links is chosen. The
feedback channels are **never** jammed, so feedback stays reliable (as in the
AC-RLNC / SR-ARQ models).

- `jammer_k` (`k`) is the jamming *intensity*, swept `0…P·H`: at `k = 0` the
  network is the plain (non-jammed) one, and at `k = P·H` every link is blocked
  (throughput → 0).
- `jammer_alpha` (`α`) is the jamming-block duration parameter: A block lasts
  `RTT/α` slots, so larger `α` ⇒ shorter blocks.

### Network Architecture

Both jammed networks share one topology: `P` **independent single-path chains** of
`H` hops, with a `Jammer` over all `P·H` forward links.

```text
Jammed network — P = 4 independent chains × H = 3 hops  (RTT = 12)

                     hop 1         hop 2         hop 3
                 ┌─────── node ─────── node ───────┐
   sender ───────┼─────── node ─────── node ───────┼──▶ receiver
                 ├─────── node ─────── node ───────┤
                 └─────── node ─────── node ───────┘

   Jammer: blocks k of the P·H forward links, re-chosen at random every RTT/α slots
           (the end-to-end feedback channels are never jammed)
```

- **AC-RLNC** (`JamMpMhNetwork`): `SimSender` → one `Node` per (chain, hop) →
  `SimReceiver`. It subclasses `MhNetwork`, **not** `MpMhNetwork`: Each `Node` is
  **single-in / single-out**, so the chains are fixed and isolated: there is **no
  cross-chain mixing, no recoding across paths, and no runtime natural matching**
  (in contrast to `mp_mh_network`'s `MpMhNetwork`). The source still runs the full
  MP AC-RLNC schedule (bit-filling / FEC / FB-FEC) across the `P` chains; each relay
  just recodes on its own single path.
- **SR-ARQ** (`SRJamMpMhNetwork`): Identical topology with `SRSimSender` / `SRNode`
  / `SRSimReceiver`: Exactly the MP-MH SR-ARQ network with `JamPath`s swapped in.

So both protocols sit on the same `P`-independent-chains shape (AC-RLNC's richer
per-hop mixing is deliberately dropped here), leaving the protocol as the only
variable.

### Experiment Modes

Both drivers expose the same seven modes (set `MODE` at the top of `_run_main`).
Every mode uses `th = 0` and `ō = 2·P·(RTT − 1)` (the same sender knobs as the
MP-MH AC-RLNC simulation), with E2E feedback, `num_packets_to_send = 500` run to
completion (no horizon; see **Metrics**), and `NUM_ITERATIONS = 150`.

#### Major Modes: ε-Grid Sweeps

Each sweeps the `(ε₁, ε₂)` 8×8 grid (`np.arange(0.1, 0.9, 0.1)`, as in the MP-MH
drivers) and plots one 3-D surface per series value on each metric subplot.

**`sweep_eps_grid_per_k`:** One surface per jammer intensity `k`. The `k = 0`
surface is directly comparable to the non-jammed MP-MH plot.

| Parameter | Value |
| --- | --- |
| `RTT` | 12 |
| `P` (num_paths) | 4 |
| `H` (num_hops) | 3 |
| `th` (threshold) | 0 |
| `ō` (max_allowed_overlap) | `2·P·(RTT − 1)` = 88 |
| `jammer_k` | 0, 3, 6, 9 (one surface each) |
| `jammer_alpha` | 1 |
| `num_packets_to_send` | 500 |
| `max_iterations` | `None` (no horizon) |
| `NUM_ITERATIONS` | 150 |

**`sweep_eps_grid_per_alpha`:** One surface per jammer tempo `α` (`k` fixed).

| Parameter | Value |
| --- | --- |
| `RTT` | 12 |
| `P` (num_paths) | 4 |
| `H` (num_hops) | 3 |
| `th` (threshold) | 0 |
| `ō` (max_allowed_overlap) | `2·P·(RTT − 1)` = 88 |
| `jammer_k` | 3 |
| `jammer_alpha` | 12, 1, 0.5, 0.1 (one surface each) |
| `num_packets_to_send` | 500 |
| `max_iterations` | `None` (no horizon) |
| `NUM_ITERATIONS` | 150 |

**`sweep_eps_grid_per_p`:** One surface per channel count `P`; the ε matrix is the
paper 4×3 template tiled cyclically to `P×H`.

| Parameter | Value |
| --- | --- |
| `RTT` | 12 |
| `P` (num_paths) | 4, 8, 12, 16 (one surface each) |
| `H` (num_hops) | 3 |
| `th` (threshold) | 0 |
| `ō` (max_allowed_overlap) | `2·P·(RTT − 1)` = 88 / 176 / 264 / 352 |
| `jammer_k` | 3 |
| `jammer_alpha` | 1 |
| `num_packets_to_send` | 500 |
| `max_iterations` | `None` (no horizon) |
| `NUM_ITERATIONS` | 150 |

**`sweep_eps_grid_per_h`:** One surface per hop count `H`. Uses `RTT = 72` so
`RTT/2` is divisible by every `H`.

| Parameter | Value |
| --- | --- |
| `RTT` | **72** |
| `P` (num_paths) | 4 |
| `H` (num_hops) | 3, 6, 9, 12 (one surface each) |
| `th` (threshold) | 0 |
| `ō` (max_allowed_overlap) | `2·P·(RTT − 1)` = 568 |
| `jammer_k` | 3 |
| `jammer_alpha` | 1 |
| `num_packets_to_send` | 500 |
| `max_iterations` | `None` (no horizon) |
| `NUM_ITERATIONS` | 150 |

#### Minor Modes: Line Sweeps & Sanity Check

These reuse the base settings above (`RTT` 12, `P` 4, `H` 3, `th` 0, `ō` 88, 500
packets, 150 iterations); only the jammer settings / ε differ.

- **`sweep_k`:** Sweep `jammer_k` over `0…P·H` at a fixed `(ε₁, ε₂)` and `α = 2`,
  plotting throughput / mean delay / max delay vs `k`.
- **`sweep_k_multi_eps`:** Same, overlaying one line per `(ε₁, ε₂)` pair.
- **`validate_k0`:** Sanity check. At `k = 0`, compare the jammed network to the
  plain `MpMhNetwork` / `SRMpMhNetwork` on the same ε matrix (chains vs layered
  cascade), printing mean ± std side-by-side.

### Metrics

Same `SimulationStats` pipeline as the other networks (normalized throughput,
mean / max in-order delay).

**Horizon vs. run-to-completion.** Each driver exposes `MAX_ITERATIONS`:
- `MAX_ITERATIONS = None` runs **to completion**: It continues until all
  `num_packets_to_send` packets are delivered in order, giving uncensored
  throughput and delay. This is the **preferred and reported** setting. Caveat: a
  point that can never reach the target (e.g. `k = P·H`, or very high `ε` with large
  `k`) will **not terminate**, so only sweep points that stay deliverable.
- A finite `MAX_ITERATIONS` is a **measurement horizon**: The run stops at that
  slot even if the packet target was not met. Throughput is snapshotted there and
  the in-order delays are **right-censored** (packets still in flight are not
  counted), which understates delay at heavily-jammed points. Its one advantage is
  guaranteed termination, useful when a point is unreachable (e.g. large `k`).

### SR-ARQ Jamming Parameters

SR-ARQ runs the same seven modes with the same per-mode jammer / topology sweeps
(`jammer_k`, `jammer_alpha`, `P`, `H`, `RTT` exactly as in the grid tables above).
It has no `th` / `ō`; its protocol knobs are:

| Parameter | Value |
| --- | --- |
| `feedback_mode` | `E2E_FORWARD_ONLY` |
| `window` (`SR_WINDOW`) | `2·(RTT − 1)` = 22 (end-to-end-RTT window, per chain) |
| `in_order_forwarding` | `False` |
| `node_queue_size` | `None` (unbounded) |
| `packets_per_path` | `None` |
| `num_packets_to_send` | 500 |
| `max_iterations` | `None` (no horizon) |
| `NUM_ITERATIONS` | 150 |

### Running

```bash
python scripts/jam_mp_mh_simulation.py   # AC-RLNC under jamming
python scripts/sr_arq_jam_simulation.py  # SR-ARQ under jamming
```

Pick the experiment with `MODE`, and the jammer with `SWEEP_K_ALPHA` / `K_VALUES` /
`ALPHA_VALUES` / `SWEEP_PH_K` / `P_VALUES` / `H_VALUES` etc., all near the top of
`_run_main`. (Feedback is fixed to E2E, as above.) Both honor `SIM_PARALLEL_WORKERS`
(set by the SLURM sbatch script) and `LOAD_EXISTING = True` to replot from the
pickle. Output filenames are tagged with the mode, feedback mode, and knobs so runs
never collide. (SR-ARQ defaults to `LOW_MEMORY = True` to survive long,
heavily-jammed runs.)

### Output Files & Re-Plotting

- Each mode writes its own `.pkl` + `.png`, tagged by feedback mode and knobs —
  e.g. `jam_sweep_k_results_E2E.pkl`, `jam_sweep_eps_grid_per_k_results_E2E.pkl`;
  SR-ARQ adds the window / forwarding / horizon tags
  (`sr_jam_sweep_eps_grid_per_k_results_E2E_FORWARD_ONLY_window_22_RTT_12_…`).
- `python scripts/plot_saved_results.py <pkl> […]` recognizes the per-k / per-alpha
  / per-P / per-H surface pickles from the filename and overlays one surface per
  series value; `--paper` gives the stacked figure.
- `python scripts/analyze_results_gui.py` opens the sortable per-(file × ε) table
  (one tab per `k` / `α`), with the optional unified `goodput` column for
  cross-protocol comparison.
- `python scripts/combine_sr_jam_eps_grid_per_k.py` stitches the split SR-ARQ
  `sweep_eps_grid_per_k` partial pickles (run per-`k` under SLURM) back into one
  `dict{k: […]}` under `results/`.



