"""Parity + bounded-memory tests for the SR-ARQ low-memory mode (`low_memory` flag).

The low-memory mode only skips write-only per-packet history logging (channel /
dropped / jammed / received / source-feedback histories) and prunes the per-path
slot->seq record; it never changes control flow or the RNG draw order. So:

- LOWMEM1: with the same seed, low_memory=True must produce byte-identical
  SimulationStats to low_memory=False (the counters feeding the stats are kept).
- LOWMEM2: after a heavy k=9 run, every unbounded per-packet history is empty
  while the always-on counters are positive, and creation_time_to_seq is pruned
  to the sender's ~2*RTT window.

Run: python tests/test_sr_arq_low_memory.py
"""

import os
import random
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "jamming_simulation"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sr_arq"))

from sr_arq.SRJamNetwork import SRJamMpMhNetwork
from sr_arq.sr_feedback import SRFeedbackMode


SEED = 20240925
P, H = 4, 3
GLOBAL_PROP_DELAY = 6                 # RTT/2 for RTT=12; divisible by H=3
RTT = 2 * GLOBAL_PROP_DELAY
WINDOW = 2 * (RTT - 1)                 # E2E window sizing (matches the driver)
JAMMER_ALPHA = 2
NUM_PACKETS = 100
FEEDBACK_MODE = SRFeedbackMode.E2E_FORWARD_ONLY  # the driver's mode
# Paper-ish lossy 4x3 grid so there are genuine erasures (num_dropped > 0).
EPS = [[0.1, 0.2, 0.1], [0.2, 0.1, 0.2], [0.1, 0.3, 0.1], [0.2, 0.1, 0.3]]

# Stat fields that must match exactly between the two modes.
_STAT_FIELDS = (
    "normalized_throughput",
    "inorder_delay_mean",
    "inorder_delay_max",
    "time_slots",
    "num_information_packets_decoded",
    "num_information_packets_sent",
    "num_transmissions",
    "num_new_rlnc_packets",
    "num_fec_packets",
    "num_fb_fec_packets",
    "num_transmissions_dropped",
)


def _build(low_memory: bool, jammer_k: int, max_iterations: int) -> SRJamMpMhNetwork:
    return SRJamMpMhNetwork(
        path_epsilons=EPS,
        num_paths=P,
        num_hops=H,
        global_prop_delay=GLOBAL_PROP_DELAY,
        num_packets_to_send=NUM_PACKETS,
        max_iterations=max_iterations,
        window=WINDOW,
        feedback_mode=FEEDBACK_MODE,
        jammer_alpha=JAMMER_ALPHA,
        jammer_k=jammer_k,
        low_memory=low_memory,
    )


def _run(low_memory: bool, jammer_k: int, max_iterations: int) -> SRJamMpMhNetwork:
    # Re-seed before each build+run so both modes draw the identical RNG sequence
    # (channel erasures + jammer path selection); low_memory changes neither.
    random.seed(SEED)
    net = _build(low_memory, jammer_k, max_iterations)
    net.run_sim()
    return net


def _all_paths(net):
    return [p for chain in net.paths for p in chain]


def test_low_memory_matches_full_stats():
    """Same seed, light jamming (decodes the target): low_memory must reproduce the
    full-history run's SimulationStats field-for-field."""
    print("\n=== LOWMEM1: low_memory=True reproduces low_memory=False stats ===")
    jammer_k, max_iterations = 4, 5000
    full = _run(low_memory=False, jammer_k=jammer_k, max_iterations=max_iterations).get_simulation_stats()
    low = _run(low_memory=True, jammer_k=jammer_k, max_iterations=max_iterations).get_simulation_stats()

    for field in _STAT_FIELDS:
        assert getattr(full, field) == getattr(low, field), (
            f"stat '{field}' differs: full={getattr(full, field)} low={getattr(low, field)}"
        )
    # Non-trivial run: something decoded (so the collect_stats override actually ran).
    assert low.num_information_packets_decoded > 0, "expected some deliveries at k=4"
    assert low.num_transmissions > 0
    print(
        f"  decoded={low.num_information_packets_decoded} tp={low.normalized_throughput:.4f} "
        f"n_tx={low.num_transmissions} dropped={low.num_transmissions_dropped}  PASSED"
    )


def test_low_memory_bounds_history():
    """Heavy k=9 run: every unbounded per-packet history is empty while the
    always-on counters are positive, and creation_time_to_seq is pruned."""
    print("\n=== LOWMEM2: low_memory=True keeps per-packet histories bounded (k=9) ===")
    net = _run(low_memory=True, jammer_k=9, max_iterations=3000)

    total_jammed = 0
    total_dropped = 0
    for path in _all_paths(net):
        fc = path.forward_channel
        assert fc.channel_history == [], "forward channel_history must be empty in low-memory"
        assert fc.dropped_packets == [], "dropped_packets list must be empty in low-memory"
        assert fc.jammed_packets_history == [], "jammed_packets_history must be empty in low-memory"
        assert path.feedback_channel.channel_history == [], "feedback channel_history must be empty"
        total_jammed += fc.num_jammed
        total_dropped += fc.num_dropped
    assert total_jammed > 0, "k=9 must jam packets (num_jammed counter should be > 0)"

    # E2E feedback channels also skip their history log.
    for ch in net.e2e_feedback_channels.values():
        assert ch.channel_history == [], "E2E feedback channel_history must be empty"

    # Source: object list empty, counter positive; source feedback log empty.
    assert net.sender.sent_new_rlnc_history == [], "sent_new_rlnc_history must be empty in low-memory"
    assert net.sender.num_new_transmissions > 0, "num_new_transmissions counter should be > 0"
    assert net.sender.all_feedback_history == [], "source all_feedback_history must be empty (E2E)"

    # Final receiver + every relay receiver: received / sent-feedback logs empty.
    assert net.receiver.received_rlnc_channel_history == []
    assert net.receiver.sent_feedback_channel_history == []
    for chain in net.nodes:
        for node in chain:
            assert node.my_receiver.received_rlnc_channel_history == [], (
                "relay received_rlnc_channel_history must be empty in low-memory"
            )

    # creation_time_to_seq is pruned to the sender's window (~2 * e2e RTT).
    bound = net.sender._prune_window + 1
    for p in net.sender.paths:
        assert len(p.creation_time_to_seq) <= bound, (
            f"creation_time_to_seq not pruned: {len(p.creation_time_to_seq)} > {bound}"
        )
    print(
        f"  jammed={total_jammed} dropped={total_dropped} "
        f"n_tx={net.sender.num_new_transmissions} prune_bound={bound}  PASSED"
    )


def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING SR-ARQ LOW-MEMORY TESTS")
    print("=" * 70)
    test_low_memory_matches_full_stats()
    test_low_memory_bounds_history()
    print("\n" + "=" * 70)
    print("ALL SR-ARQ LOW-MEMORY TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
