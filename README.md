# MP MH AC-RLNC Simulation

## SR ARQ Simulation

An uncoded Selective-Repeat ARQ (SR-ARQ) protocol simulation, used as a baseline
to compare against the adaptive-causal network coding protocols. It runs on the
same discrete-time, BEC forward-channel + reliable-feedback model, so results are
directly comparable.

Everything for SR-ARQ lives in `sr_arq/`, built on top of the shared
`GeneralSender` / `GeneralReceiver` / `Path` / `Channel` machinery.

### What SR-ARQ does here

- Uncoded: each data packet carries exactly one information packet (its sequence
number, `seq`). Feedback is ACK / NACK only, one per received / missed slot.
- NACK-driven retransmission: a lost packet is detected by the built-in per-slot
NACK and retransmitted. Feedback is lossless, so no timer is used. A NACK is
slot-based `(path, creation_time)`; the sender maps it back to a `seq` via a
per-path `creation_time -> seq` record. ACKs carry the `seq` directly.
- In-order delivery is enforced at the final receiver. Intermediate nodes
forward out-of-order by default (send the lowest available not-yet-forwarded
seq, skip gaps; the gap fills later via the upstream link's own retransmission).
An `in_order_forwarding=True` option makes each node a full per-hop SR-ARQ
endpoint (release only the contiguous in-order prefix), adding per-hop
head-of-line blocking and higher delay.
- Sliding window (flow control, not reliability): caps the number of outstanding
(sent-but-not-ACKed) packets per chain. A finite window throttles throughput
below the raw link rate (like real SR-ARQ); `window=None` is unbounded and
approaches capacity.



### Two SR-ARQ variants

- Shared multipath (`SRSender` + `SRReceiver`): one global stream striped across
all paths, with retransmissions rerouted onto whichever path is free, and
global in-order delivery. This is a stronger, non-standard multipath design.
- Decoupled per-chain (`SRSimSender` + `SRSimReceiver`): each path/chain is an
independent SR-ARQ flow on a static round-robin slice of the sequence space,
with no rerouting and per-chain in-order delivery at the receiver. A bad chain
never blocks a good one. This is the standard "SR-ARQ applied independently on
each path" baseline.



### Components (`sr_arq/`)

- `SRPacket.py` - `DataPacket` (uncoded, single `seq`) and `SRType`.
- `SRSender.py` - `SRSender` (shared multipath) and `SRSimSender` (decoupled
per-chain, per-chain sliding window).
- `SRReceiver.py` - `SRReceiver` (global in-order, single-hop) and
`SRSimReceiver` (decoupled per-chain in-order, records per-chain finish times
and per-seq delivery times).
- `SRNode.py` - hop-by-hop relay: `SRNodeReceiver` (input link: ACK/NACK
upstream, buffer received seqs; `in_order_forwarding` option), `SRNodeSender`
(output link: forward buffered seqs, retransmit on downstream NACK), and
`SRNode` wiring them.
- `SRNetwork.py`:
  - `SRNetwork` - single-hop, P paths (shared or decoupled via `independent`).
  - `SRMpMhNetwork` - multi-hop, P independent chains of H hops with one
  `SRNode` per (chain, hop); source on hop 0, `SRSimReceiver` on the last hop;
  explicit per-slot tick (source -> nodes hop-major -> receiver); `window` and
  `in_order_forwarding` options. H=1 reduces to the single-hop decoupled model.



### Metrics

- Normalized throughput: information packets delivered in order per time slot.
For the decoupled model it is the sum of per-chain rates
(`delivered_c / finish_time_c`), so a slow chain does not drag down the others.
- In-order delivery delay (mean and max): `delivery_time - first_source_transmission_time`
(ACK propagation not counted).



### Running

`scripts/sr_arq_simulation.py` sweeps the paper's MP setting (P=4, RTT=20,
eps_3=0.2, eps_4=0.8, eps_1/eps_2 in [0.1, 0.8]) over many channel realizations
and plots throughput / mean delay / max delay surfaces.

Select which protocol(s) to run via the `protos` list in `_run_main`:
`"sr"` (shared), `"sr_indep"` (coupled round-robin), `"sr_perpath"` (decoupled
single-hop), `"sr_mpmh"` (decoupled multi-hop). Key knobs: `NUM_HOPS`,
`SR_WINDOW`, `NUM_PACKETS_TO_SEND`, `NUM_ITERATIONS`, `PARALLEL_WORKERS`.

```
python scripts/sr_arq_simulation.py
```

`scripts/sr_arq_mpmh_simulation.py` reproduces the paper's Fig. 19 lower-graph
MP-MH setting (H=3, P=4, RTT=12, the paper's 4x3 epsilon matrix) in two
settings: a single global path built from the best path of each hop, and the P
natural-matched global paths. Knobs: `IN_ORDER_FORWARDING`, `SR_WINDOW`,
`NUM_PACKETS_TO_SEND`, `NUM_ITERATIONS`.

```
python scripts/sr_arq_mpmh_simulation.py
```

### Status

Done:
- Single-hop MP SR-ARQ (shared + decoupled per-path): throughput rises with
  channel quality, sits below AC-RLNC, and `D_max >> D_mean` - matches the
  paper's Fig. 11 shape.
- Multi-hop hop-by-hop SR-ARQ (`SRMpMhNetwork`), decoupled per-chain metrics;
  H=1 reduces to (and matches) the single-hop model.
- MP-MH paper script with the two Fig. 19 settings (best single path, P matched
  paths via natural matching).
- Sliding-window flow control; NACK-driven retransmission; node forwarding both
  out-of-order (default) and in-order (option).
- MP-MH matched setting is close to the paper's stated targets (mean ~25x, max
  ~84x the genie bound), matching well at high loss (e.g. eps=0.8: mean 283 vs
  233, max 697 vs 784).

Current bugs / open discrepancies:
- Best single-path delay is much lower than the paper's Fig. 19 curve appears to
  be. Unconfirmed: the paper gives no numeric single-path target, and a
  best-per-hop chain is expected to be low-delay. Needs the paper's actual
  single-path values to judge.
- At low loss the matched delays run ~0.6x the paper's targets (closer at high
  loss).
- Throughput and delay are coupled through the single `window` parameter, so one
  window cannot independently match both the paper's throughput and its delay;
  the window is currently calibrated to throughput (`RTT-1`).
- `D_max` is the global worst packet over all paths; the paper's `D_max` bound
  uses an average/virtual-path notion, so the metric definitions may differ.

What's left to finish SR-ARQ:
- Validate the single best-path MP-MH setting against the paper's actual Fig. 19
  values.
- Calibrate the `window` (and/or `in_order_forwarding`) to match the paper's
  absolute delays.
- Jammer: add `SRJamNetwork` (using `JamPath` + `Jammer`) and jammed sweeps.
- Wire an AC-RLNC (`MpMhNetwork`) overlay into the SR drivers for a direct
  side-by-side (note the epsilon-matrix convention: `MpMhNetwork` is
  `[hop][path]`, the SR code is chain-major `[chain][hop]`).

