"""
End-to-end tests for MpMhNetwork.

These tests build a full multi-path multi-hop network
(SimSender → Node(s) → SimReceiver) and run the simulation
via run_sim().  They verify that all information packets are
decoded by the SimReceiver.

Test naming convention:
- MH1: 3 hops, 4 paths, eps=0 — baseline, all 100 packets decoded
- MH2: 3 hops, 4 paths, eps=0.1 — light loss, all 100 packets decoded
- MH3: Verify the full chain structure (sender → nodes → receiver)
- MH4: 2 hops (1 node), 4 paths, eps=0 — minimal multi-hop
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Network import MpMhNetwork


# ============================================================================
# MH1 — 3 hops, 4 paths, eps=0, 100 packets
# ============================================================================

def test_MH1_zero_loss_3_hops_4_paths():
    """Baseline: no erasures anywhere.  All 100 information packets must
    be decoded by the SimReceiver."""
    NUM_PACKETS = 100
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 2

    print(f"\n=== Test MH1: {NUM_HOPS} hops, {NUM_PATHS} paths, eps=0, "
          f"{NUM_PACKETS} packets ===")

    epsilons = [[0.0] * NUM_PATHS for _ in range(NUM_HOPS)]

    net = MpMhNetwork(
        path_epsilons=epsilons,
        num_packets_to_send=NUM_PACKETS,
        num_paths=NUM_PATHS,
        prop_delay=PROP_DELAY,
        num_hops=NUM_HOPS,
        # debug=False,
    )

    net.run_sim()
    stats = net.get_simulation_stats()

    expected_packets = set(range(1, NUM_PACKETS + 1))

    decoded_packets = set(net.receiver.information_packets_decoding_times.keys())
    assert decoded_packets == expected_packets, \
        f"Decoded packets mismatch.\n  Missing: {expected_packets - decoded_packets}\n  Extra:   {decoded_packets - expected_packets}"

    sent_packets = set(net.sender.inforamtion_packets_first_transmission_times.keys())
    assert expected_packets == sent_packets, \
        f"Sender should have sent all {NUM_PACKETS} info packets.\n  Missing: {expected_packets - sent_packets}"

    assert stats.num_transmissions_dropped == 0, \
        f"No packets should be dropped with eps=0, got {stats.num_transmissions_dropped}"
    assert stats.normalized_throughput > 0, "Throughput should be positive"

    print(f"  Decoded:    {decoded_packets == expected_packets}")
    print(f"  Throughput: {stats.normalized_throughput:.3f}")
    print(f"  Sent:       {len(sent_packets)} info packets")
    print(f"  Time:       {net.t} steps")
    print("  PASSED")


# ============================================================================
# MH2 — 3 hops, 4 paths, eps=0.1, 100 packets
# ============================================================================

def test_MH2_light_loss_3_hops_4_paths():
    """50% erasures probability on every hop. The AC-RLNC protocol should still
    deliver all 100 packets (FEC + FB-FEC compensate for losses)."""
    NUM_PACKETS = 200
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6

    print(f"\n=== Test MH2: {NUM_HOPS} hops, {NUM_PATHS} paths, eps=0.5, "
          f"{NUM_PACKETS} packets ===")

    seed = random.randint(1, 50)
    # random.seed(seed)
    random.seed(29)
    print(f"Seed: {seed}")
    epsilons = [[0.5] * NUM_PATHS for _ in range(NUM_HOPS)]
    # Debug prints happen in run_sim(), not in MpMhNetwork(); redirect must wrap run_sim().
    _log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_MH2_light_loss_3_hops_4_paths.log",
    )
    from contextlib import redirect_stdout
    with open(_log_path, "w", encoding="utf-8") as _log_f, redirect_stdout(_log_f):
        net = MpMhNetwork(
            path_epsilons=epsilons,
            num_packets_to_send=NUM_PACKETS,
            num_paths=NUM_PATHS,
            prop_delay=PROP_DELAY,
            num_hops=NUM_HOPS,
            debug=True,
        )
        net.run_sim()
    stats = net.get_simulation_stats()

    expected_packets = set(range(1, NUM_PACKETS + 1))

    decoded_packets = set(net.receiver.information_packets_decoding_times.keys())
    assert decoded_packets == expected_packets, \
        f"Decoded packets mismatch.\n  Missing: {expected_packets - decoded_packets}\n  Extra:   {decoded_packets - expected_packets}"

    # Every decoded packet must have a valid decoding time > its sending time
    for pkt, decode_time in net.receiver.information_packets_decoding_times.items():
        send_time = net.sender.inforamtion_packets_first_transmission_times[pkt]
        assert decode_time > send_time, \
            f"Packet {pkt}: decode_time ({decode_time}) must be > send_time ({send_time})"

    print(f"  Decoded:    {decoded_packets == expected_packets}")
    print(f"  Throughput: {stats.normalized_throughput:.3f}")
    print(f"  Dropped:    {stats.num_transmissions_dropped}")
    print(f"  Time:       {net.t} steps")
    print("  PASSED")


# ============================================================================
# MH3 — Verify the chain structure
# ============================================================================

def test_MH3_chain_structure():
    """Build a 3-hop network and verify the wiring:
    SimSender.next_hop → Node[0] → Node[1] → SimReceiver.
    Also verify that after run_sim, every node received and forwarded packets,
    and the receiver decoded the exact expected set."""
    NUM_PACKETS = 50
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 2

    print(f"\n=== Test MH3: chain structure verification ===")

    epsilons = [[0.0] * NUM_PATHS for _ in range(NUM_HOPS)]

    net = MpMhNetwork(
        path_epsilons=epsilons,
        num_packets_to_send=NUM_PACKETS,
        num_paths=NUM_PATHS,
        prop_delay=PROP_DELAY,
        num_hops=NUM_HOPS,
        max_iterations=500,
    )

    # --- Verify wiring before running ---
    assert len(net.nodes) == NUM_HOPS - 1, \
        f"Expected {NUM_HOPS - 1} nodes, got {len(net.nodes)}"

    assert net.sender.next_hop is net.nodes[0], \
        "SimSender.next_hop should be Node[0]"
    assert net.nodes[0].next_hop is net.nodes[1], \
        "Node[0].next_hop should be Node[1]"
    assert net.nodes[1].next_hop is net.receiver, \
        "Node[1].next_hop should be SimReceiver"

    for hop_idx in range(NUM_HOPS):
        assert len(net.paths[hop_idx]) == NUM_PATHS, \
            f"Hop {hop_idx} should have {NUM_PATHS} paths"

    # --- Run simulation ---
    net.run_sim()

    expected_packets = set(range(1, NUM_PACKETS + 1))

    # --- Verify every node participated ---
    for i, node in enumerate(net.nodes):
        node_received = set(
            pkt.get_global_path() for pkt in node.my_receiver.get_received_rlnc_channel_history()
        )
        assert len(node_received) > 0, \
            f"Node[{i}] receiver should have received packets on multiple global paths"

        node_sent_new = node.my_sender.new_rlnc_packets_history
        node_sent_corr = node.my_sender.correction_packets_history
        assert len(node_sent_new) + len(node_sent_corr) > 0, \
            f"Node[{i}] sender should have forwarded packets"

        node_new_info = node.my_sender.new_information_packets_history
        assert len(node_new_info) > 0, \
            f"Node[{i}] should have buffered new information packets"

        node_fb = sum(len(p.all_feedback_history) for p in node.my_sender.paths)
        assert node_fb > 0, \
            f"Node[{i}] sender should have received feedback"

        print(f"  Node[{i}]: rx_paths={node_received}, "
              f"sent={len(node_sent_new)}+{len(node_sent_corr)}, "
              f"new_info={len(node_new_info)}, fb={node_fb}")

    # --- Verify receiver decoded the exact set ---
    decoded_packets = set(net.receiver.information_packets_decoding_times.keys())
    assert decoded_packets == expected_packets, \
        f"Decoded packets mismatch.\n  Missing: {expected_packets - decoded_packets}\n  Extra:   {decoded_packets - expected_packets}"

    print("  PASSED")


# ============================================================================
# MH4 — 2 hops (1 node), 4 paths, eps=0
# ============================================================================

def test_MH4_two_hops_one_node():
    """Minimal multi-hop: SimSender → Node[0] → SimReceiver.
    Verifies the 2-hop case works correctly."""
    NUM_PACKETS = 100
    NUM_PATHS = 4
    NUM_HOPS = 2
    PROP_DELAY = 2

    print(f"\n=== Test MH4: {NUM_HOPS} hops, {NUM_PATHS} paths, eps=0, "
          f"{NUM_PACKETS} packets ===")

    epsilons = [[0.0] * NUM_PATHS for _ in range(NUM_HOPS)]

    net = MpMhNetwork(
        path_epsilons=epsilons,
        num_packets_to_send=NUM_PACKETS,
        num_paths=NUM_PATHS,
        prop_delay=PROP_DELAY,
        num_hops=NUM_HOPS,
        max_iterations=1000,
    )

    assert len(net.nodes) == 1, f"Expected 1 node, got {len(net.nodes)}"
    assert net.sender.next_hop is net.nodes[0]
    assert net.nodes[0].next_hop is net.receiver

    net.run_sim()

    expected_packets = set(range(1, NUM_PACKETS + 1))

    decoded_packets = set(net.receiver.information_packets_decoding_times.keys())
    assert decoded_packets == expected_packets, \
        f"Decoded packets mismatch.\n  Missing: {expected_packets - decoded_packets}\n  Extra:   {decoded_packets - expected_packets}"

    sent_packets = set(net.sender.inforamtion_packets_first_transmission_times.keys())
    assert expected_packets.issubset(sent_packets), \
        f"Sender should have sent all info packets.\n  Missing: {expected_packets - sent_packets}"

    # The single node must have received and forwarded
    node = net.nodes[0]
    node_new_info = node.my_sender.new_information_packets_history
    assert expected_packets.issubset(node_new_info), \
        f"Node new info buffer should contain all sent info packets.\n  Missing: {expected_packets - node_new_info}"

    print(f"  Decoded:    {decoded_packets == expected_packets}")
    print(f"  Throughput: {net.get_simulation_stats().normalized_throughput:.3f}")
    print(f"  Time:       {net.t} steps")
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING MP-MH NETWORK TESTS")
    print("=" * 70)

    test_MH1_zero_loss_3_hops_4_paths()
    test_MH2_light_loss_3_hops_4_paths()
    test_MH3_chain_structure()
    test_MH4_two_hops_one_node()

    print("\n" + "=" * 70)
    print("ALL MP-MH NETWORK TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
