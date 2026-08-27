"""Smoke test for JamMpMhNetwork.

Builds a small JamMpMhNetwork (H=3, P=3) with a Jammer that blocks 1 of 9
forward channels per round. Checks:
  1. Construction succeeds.
  2. run_sim() terminates and populates simulation_stats with non-zero
     throughput (since we jam only 1/9 paths and have plenty of redundancy).
  3. Per-chain isolation: each Node's my_receiver buffer only holds info-ids
     that arrived on that chain (trivially true because each Node has 1
     input path -- this just sanity-checks the wiring).
  4. Jammer state: jammed_paths_history populated.
  5. JamForwardChannel state: jammed_packets_history populated for at least
     one path.
  6. Edge case: jamming ALL 9 paths produces zero decoded packets within
     a short iteration budget (the network cannot make forward progress).
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "jamming_simulation"))

from JamNetwork import JamMpMhNetwork
from feedback_source import FeedbackSource


def _uniform_eps(num_paths: int, num_hops: int, eps: float) -> list[list[float]]:
    """Build a chain-major path_epsilons[c][h] matrix with uniform epsilon."""
    return [[eps for _ in range(num_hops)] for _ in range(num_paths)]


def smoke_basic():
    print("\n=== Smoke 1: H=3, P=3, packets=50, k=1 (light jamming) ===")
    H, P = 3, 3
    NUM_PACKETS = 50
    PROP_DELAY = 6
    network = JamMpMhNetwork(
        path_epsilons=_uniform_eps(P, H, eps=0.1),
        num_paths=P,
        num_hops=H,
        global_prop_delay=PROP_DELAY,
        num_packets_to_send=NUM_PACKETS,
        max_iterations=4000,
        jammer_alpha=2,
        jammer_k=1,
        debug=False,
    )

    assert len(network.paths) == P, f"paths chain count {len(network.paths)} != P={P}"
    for c, chain in enumerate(network.paths):
        assert len(chain) == H, f"chain {c} has {len(chain)} hops, expected H={H}"
    assert len(network.nodes) == P, f"nodes chain count {len(network.nodes)} != P={P}"
    for c in range(P):
        assert len(network.nodes[c]) == H - 1, f"chain {c} has {len(network.nodes[c])} nodes, expected H-1={H-1}"
    print(f"  Topology OK: paths[{P}][{H}], nodes[{P}][{H-1}]")

    network.run_sim()

    stats = network.get_simulation_stats()
    assert stats is not None, "simulation_stats should be populated after run_sim"
    assert stats.normalized_throughput > 0.0, f"throughput should be >0, got {stats.normalized_throughput}"
    assert stats.num_information_packets_decoded == NUM_PACKETS, (
        f"expected {NUM_PACKETS} decoded, got {stats.num_information_packets_decoded}"
    )
    print(
        f"  decoded={stats.num_information_packets_decoded}/{NUM_PACKETS} "
        f"in t={network.t}, throughput={stats.normalized_throughput:.3f}, "
        f"mean delay={stats.inorder_delay_mean:.2f}, max delay={stats.inorder_delay_max}"
    )

    jammed_history = network.jammer.get_jammed_paths_history()
    assert len(jammed_history) > 0, "jammer should have recorded at least one resample"
    print(f"  jammer rounds recorded: {len(jammed_history)}")

    total_jammed_pkts = sum(
        len(p.forward_channel.jammed_packets_history)
        for chain in network.paths
        for p in chain
    )
    assert total_jammed_pkts > 0, "expected at least one packet to be jammed"
    print(f"  total jammed packets across all paths: {total_jammed_pkts}")

    for c in range(P):
        for h in range(H - 1):
            node = network.nodes[c][h]
            chain_gpi = c + 1
            buf = node.my_receiver.new_information_packets_buffer
            assert chain_gpi in buf or len(buf) == 0, (
                f"Node[c={c},h={h}] new_information_packets_buffer keys "
                f"{list(buf.keys()) if isinstance(buf, dict) else 'set'} "
                f"unexpected for chain {c} (gpi={chain_gpi})"
            )
    print("  per-chain isolation OK")
    print("  PASSED")


def smoke_full_jam():
    print("\n=== Smoke 2: H=2, P=3, k=6 (jam ALL 6 paths) -> 0 decoded ===")
    # Iterate _tick directly to avoid the pre-existing ZeroDivisionError in
    # Network.collect_stats() when no packets decode.
    H, P = 2, 3
    NUM_PACKETS = 20
    NUM_STEPS = 100
    network = JamMpMhNetwork(
        path_epsilons=_uniform_eps(P, H, eps=0.0),
        num_paths=P,
        num_hops=H,
        global_prop_delay=4,
        num_packets_to_send=NUM_PACKETS,
        max_iterations=NUM_STEPS,
        jammer_alpha=1,
        jammer_k=P * H,
        debug=False,
    )

    for t in range(1, NUM_STEPS + 1):
        network.t = t
        network._tick()

    decoded = len(network.receiver.information_packets_decoding_times)
    assert decoded == 0, f"With all paths jammed, expected 0 decoded, got {decoded}"

    total_jammed_pkts = sum(
        len(p.forward_channel.jammed_packets_history)
        for chain in network.paths
        for p in chain
    )
    assert total_jammed_pkts > 0, "expected the jammer to have blocked something"
    print(
        f"  decoded={decoded} (expected 0), total jammed packets={total_jammed_pkts}"
    )
    print("  PASSED")


def smoke_no_jam():
    print("\n=== Smoke 3: H=3, P=3, k=0 (no jamming) -> all decoded, no jam events ===")
    H, P = 3, 3
    NUM_PACKETS = 30
    network = JamMpMhNetwork(
        path_epsilons=_uniform_eps(P, H, eps=0.05),
        num_paths=P,
        num_hops=H,
        global_prop_delay=6,  # must be divisible by num_hops (H=3)
        num_packets_to_send=NUM_PACKETS,
        max_iterations=4000,
        jammer_alpha=2,
        jammer_k=0,
        debug=False,
    )

    network.run_sim()
    stats = network.get_simulation_stats()
    assert stats.num_information_packets_decoded == NUM_PACKETS, (
        f"expected {NUM_PACKETS} decoded, got {stats.num_information_packets_decoded}"
    )

    total_jammed_pkts = sum(
        len(p.forward_channel.jammed_packets_history)
        for chain in network.paths
        for p in chain
    )
    assert total_jammed_pkts == 0, f"expected 0 jammed packets with k=0, got {total_jammed_pkts}"
    print(
        f"  decoded={stats.num_information_packets_decoded}/{NUM_PACKETS}, "
        f"jammed pkts={total_jammed_pkts}, throughput={stats.normalized_throughput:.3f}"
    )
    print("  PASSED")


def smoke_e2e():
    print("\n=== Smoke 4: H=3, P=3, k=1, end-to-end (E2E) feedback ===")
    H, P = 3, 3
    NUM_PACKETS = 50
    PROP_DELAY = 6
    network = JamMpMhNetwork(
        path_epsilons=_uniform_eps(P, H, eps=0.1),
        num_paths=P,
        num_hops=H,
        global_prop_delay=PROP_DELAY,
        num_packets_to_send=NUM_PACKETS,
        max_iterations=4000,
        jammer_alpha=2,
        jammer_k=1,
        debug=False,
        feedback_source=FeedbackSource.E2E,
    )

    # One dedicated end-to-end feedback channel per global path (= per chain).
    assert network.feedback_source == FeedbackSource.E2E
    assert len(network.e2e_feedback_channels) == P, (
        f"expected {P} E2E feedback channels, got {len(network.e2e_feedback_channels)}"
    )
    assert sorted(network.e2e_feedback_channels.keys()) == list(range(1, P + 1)), (
        f"E2E channels should be keyed by global path index 1..{P}, "
        f"got {sorted(network.e2e_feedback_channels.keys())}"
    )

    network.run_sim()

    stats = network.get_simulation_stats()
    assert stats is not None, "simulation_stats should be populated after run_sim"
    assert stats.normalized_throughput > 0.0, (
        f"throughput should be >0, got {stats.normalized_throughput}"
    )
    assert stats.num_information_packets_decoded == NUM_PACKETS, (
        f"expected {NUM_PACKETS} decoded, got {stats.num_information_packets_decoded}"
    )

    # The receiver must have emitted end-to-end feedback back to the source.
    total_e2e_feedback = sum(
        len(ch.get_channel_history()) for ch in network.e2e_feedback_channels.values()
    )
    assert total_e2e_feedback > 0, "expected the receiver to send end-to-end feedback"

    print(
        f"  decoded={stats.num_information_packets_decoded}/{NUM_PACKETS} "
        f"in t={network.t}, throughput={stats.normalized_throughput:.3f}, "
        f"mean delay={stats.inorder_delay_mean:.2f}, max delay={stats.inorder_delay_max}"
    )
    print(f"  E2E feedback channels: {P} | total E2E feedback packets: {total_e2e_feedback}")
    print("  PASSED")


if __name__ == "__main__":
    smoke_basic()
    smoke_full_jam()
    smoke_no_jam()
    smoke_e2e()
    print("\nAll smoke tests passed.")
