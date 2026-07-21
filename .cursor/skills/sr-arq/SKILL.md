---
name: sr-arq
description: Guidance for the uncoded Selective-Repeat ARQ (SR-ARQ) baseline simulation in this repo (the sr_arq/ package plus scripts/sr_arq_simulation.py and scripts/sr_arq_mpmh_simulation.py). Covers its architecture, how to run it, current status/bugs, and what is left. Use when running, extending, debugging, or reasoning about the SR-ARQ protocol simulation, or comparing it to the AC-RLNC MP / MP-MH results.
---

# SR-ARQ simulation

Uncoded Selective-Repeat ARQ baseline used to compare against the AC-RLNC
protocols. It runs on the same discrete-time BEC forward-channel + reliable
per-slot feedback model as `mp_mh_network/`, reusing `GeneralSender`,
`GeneralReceiver`, `Path`, `Channel`, and `SimulationStats`. All SR-ARQ code is
in `sr_arq/`.

## Architecture

- `sr_arq/SRPacket.py` - `DataPacket` (uncoded; one info packet = its `seq`) and `SRType`.
- `sr_arq/SRSender.py`
  - `SRSender`: shared multipath - one global seq stream striped across all P
    paths, retransmissions rerouted to any free path, global in-order at the
    receiver. Stronger than a standard SR-ARQ (our own design).
  - `SRSimSender`: decoupled per-chain - each path is an independent SR-ARQ flow
    on a static round-robin slice of the seq space (path i owns seqs
    i+1, i+1+P, ...), no rerouting, per-chain sliding window. The paper's
    "SR-ARQ applied independently on each path" baseline.
- `sr_arq/SRReceiver.py`
  - `SRReceiver`: single-hop, global in-order delivery.
  - `SRSimReceiver`: decoupled per-chain in-order (keyed by `global_path_id`);
    records per-seq delivery times and per-chain finish times.
- `sr_arq/SRNode.py` - hop-by-hop relay: `SRNodeReceiver` (input link: ACK/NACK
  upstream, buffer received seqs) + `SRNodeSender` (output link: forward seqs,
  retransmit on downstream NACK) + `SRNode`. Forwarding is out-of-order by
  default; `in_order_forwarding=True` makes each node a full per-hop SR-ARQ
  endpoint (release only the contiguous in-order prefix -> per-hop HOL delay).
- `sr_arq/SRNetwork.py`
  - `SRNetwork`: single-hop, P paths (`independent` selects shared vs decoupled).
  - `SRMpMhNetwork`: multi-hop, P independent chains of H hops, one `SRNode` per
    (chain, hop); source on hop 0, `SRSimReceiver` on the last hop; explicit
    per-slot tick (source -> nodes hop-major -> receiver). Options: `window`,
    `in_order_forwarding`. H=1 reduces to the single-hop decoupled model.

## Key protocol rules

- NACK-driven retransmission (feedback is lossless, so no timer). A NACK is
  slot-based `(global_path_id, creation_time)`; the sender maps it back to a
  `seq` via a per-path `creation_time -> seq` record. ACKs carry the `seq`.
  Process ACKs before NACKs; never re-queue an already-ACKed seq.
- Lowest-seq-first scheduling; retransmits before new seqs.
- Sliding window is flow control, not reliability: caps outstanding
  (sent-but-not-ACKed) seqs per chain. Small window throttles throughput below
  the link rate; `window=None` is unbounded (approaches capacity).
- Chain identity travels via `global_path_id` (= chain index + 1), preserved by
  nodes when they re-stamp forwarded `DataPacket`s.

## Metrics

- Normalized throughput = in-order-delivered packets per slot. Decoupled model
  uses the SUM of per-chain rates (`delivered_c / finish_time_c`), so a slow
  chain does not drag down the others.
- In-order delay (mean, max) = `delivery_time - first_source_transmission_time`
  (ACK propagation not counted). `D_max` is the single worst packet over ALL
  paths.

## Running

- `scripts/sr_arq_simulation.py` - single-hop MP setting (Fig. 11: P=4, RTT=20,
  eps_3=0.2, eps_4=0.8, eps_1/eps_2 in [0.1,0.8]). Pick protocols via `protos`
  in `_run_main`: `"sr"`, `"sr_indep"`, `"sr_perpath"`, `"sr_mpmh"`, `"ac"`.
- `scripts/sr_arq_mpmh_simulation.py` - MP-MH setting (Fig. 19 lower graph: H=3,
  P=4, RTT=12, the paper's 4x3 epsilon matrix) in two settings: `best` (single
  global path from the best path of each hop) and `matched` (P natural-matched
  global paths). Knobs: `IN_ORDER_FORWARDING`, `SR_WINDOW`, `NUM_PACKETS_TO_SEND`,
  `NUM_ITERATIONS`.
- `LOAD_EXISTING=True` replots from the pickle without re-running. Runs are heavy
  (grid x 150 iters x 4 chains); reduce iterations for a quick check.

## Status: done

- Single-hop MP SR-ARQ (shared + decoupled per-path): throughput rises with
  channel quality, sits below AC-RLNC, `D_max >> D_mean` - matches Fig. 11 shape.
- Multi-hop hop-by-hop SR-ARQ (`SRMpMhNetwork`) with decoupled per-chain metrics;
  H=1 matches the single-hop model.
- MP-MH paper script with both Fig. 19 settings (best single path, P matched).
- Sliding window; NACK-driven retransmission; out-of-order (default) and
  in-order (option) node forwarding.
- MP-MH matched is close to the paper's targets (mean ~25x, max ~84x the genie
  bound `RTT/2 + 1/(1-eps_bar)`), matching well at high loss (eps=0.8: mean 283
  vs 233, max 697 vs 784).

## Status: current bugs / open discrepancies

- Best single-path delay is much lower than the paper's Fig. 19 curve appears to
  be. Unconfirmed: the paper gives no numeric single-path target and a
  best-per-hop chain is expected to be low-delay. Needs the paper's actual
  single-path values to judge.
- At low loss the matched delays run ~0.6x the paper's targets (closer at high
  loss).
- Throughput and delay are coupled through the single `window` parameter, so one
  window cannot independently match both the paper's throughput and its delay;
  window is currently calibrated to throughput (`RTT-1`).
- `D_max` is the global worst packet; the paper's `D_max` bound uses an
  average/virtual-path notion, so the metric definitions may differ.

## Status: what is left to finish

- Validate the single best-path MP-MH setting against the paper's actual Fig. 19
  values (read them off the graph).
- Calibrate `window` (and/or `in_order_forwarding`) to the paper's absolute delays.
- Jammer: add `SRJamNetwork` (using `JamPath` + `Jammer` from
  `jamming_simulation/`) and jammed sweeps.
- Wire an AC-RLNC (`MpMhNetwork`) overlay into the SR drivers for a direct
  side-by-side. Note the epsilon-matrix convention differs: `MpMhNetwork` is
  `[hop][path]`, the SR code is chain-major `[chain][hop]`.

## Gotchas

- Do not confuse the three receiver roles: `SRReceiver` (single-hop global
  in-order), `SRSimReceiver` (final receiver, per-chain in-order + stats),
  `SRNodeReceiver` (relay input, no in-order, just buffers to forward).
- Natural matching is computed statically from the true epsilons (sort each
  hop's column ascending, match rank-to-rank).
- The paper's `25x` (mean) / `84x` (max) delay factors are for the MATCHED
  setting only; they do NOT apply to the single best path.
