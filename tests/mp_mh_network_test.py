"""
End-to-end tests for MpMhNetwork.

These tests build a full multi-path multi-hop network
(SimSender -> Node(s) -> SimReceiver) and check two things:

1. Functional correctness: run the simulation via run_sim() and verify that
   every information packet is decoded by the SimReceiver.
2. Timing: in a lossless network, verify the exact propagation timing of a
   single packet as it travels hop-by-hop, plus the per-hop feedback timing.

Timing model (verified against the implementation)
---------------------------------------------------
The whole network advances by one time step per SimSender.run_step() call, and
each forward/feedback channel carries a per-hop one-way delay of hop_prop_delay
(= global_prop_delay / num_hops = hop_rtt / 2).  A node receives on its input channel
and forwards on its output channel within the *same* tick (no extra processing
delay).  Therefore, for a packet first transmitted at t = send_time:

  - It reaches the receiver after hop `h` at:  send_time + h * hop_prop_delay
  - The final SimReceiver (after num_hops hops) gets it at:
        send_time + num_hops * hop_prop_delay
      = send_time + global_prop_delay
      = send_time + global_rtt // 2          (one-way, i.e. half the end-to-end RTT)
  - Feedback is hop-by-hop: each hop's sender hears back from its immediate
    downstream receiver after a full per-hop round trip = hop_rtt
    (= 2 * hop_prop_delay).

Test naming convention:
- MH1: 3 hops, 4 paths, eps=0   -> baseline, all packets decoded
- MH2: 3 hops, 4 paths, eps=0.1 -> light loss, all packets decoded
- MH3: verify the full chain structure (sender -> nodes -> receiver)
- MH4: 2 hops (1 node), 4 paths, eps=0 -> minimal multi-hop
- MH5: timing - lossless per-hop forward arrival (send_time + h*hop_prop_delay)
- MH6: timing - lossless per-hop feedback (ACK back at send_time + hop_rtt)
- MH7: timing - 2-hop variant, forward arrival + receiver at global_rtt//2
"""

import sys
import os
import random

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))

from Network import MpMhNetwork, FeedbackSource
from Packet import RLNCPacket, RLNCType, NodeRLNCType, FeedbackType


# ============================================================================
# Helpers
# ============================================================================

def _build_lossless_net(num_hops, num_paths, prop_delay, num_packets, max_iterations=None,
                        feedback_source=FeedbackSource.HBH):
    """Build a lossless (eps=0) MpMhNetwork."""
    epsilons = [[0.0] * num_paths for _ in range(num_hops)]
    return MpMhNetwork(
        path_epsilons=epsilons,
        num_packets_to_send=num_packets,
        num_paths=num_paths,
        global_prop_delay=prop_delay,
        num_hops=num_hops,
        max_iterations=max_iterations,
        debug=False,
        feedback_source=feedback_source,
    )


def _forward_hops(net):
    """Return an ordered list of (label, hops_traversed, receiver) for every
    receiving stage: each Node's receiver, then the final SimReceiver."""
    stages = []
    for node in net.nodes:
        # node.hop_num counts how many forward channels the packet crossed to
        # reach this node's receiver (Node[0].hop_num == 1).
        stages.append((node.unit_name, node.hop_num, node.my_receiver))
    stages.append(("SimReceiver", net.num_hops, net.receiver))
    return stages


def _first_arrival_times(receiver):
    """First-arrival time recorded on each of a receiver's input paths."""
    return [p.get_receiving_packets_strating_time() for p in receiver.receiver_paths]


# ============================================================================
# MH1 - 3 hops, 4 paths, eps=0
# ============================================================================

def test_MH1_zero_loss_3_hops_4_paths():
    """Baseline: no erasures anywhere. All information packets must be
    decoded by the SimReceiver."""
    NUM_PACKETS = 100
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6

    print(f"\n=== Test MH1: {NUM_HOPS} hops, {NUM_PATHS} paths, eps=0, "
          f"{NUM_PACKETS} packets ===")

    net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, NUM_PACKETS)
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
# MH2 - 3 hops, 4 paths, eps=0.1
# ============================================================================

def test_MH2_light_loss_3_hops_4_paths():
    """Light erasures on every hop. The AC-RLNC protocol should still deliver
    every packet (FEC + FB-FEC compensate for losses)."""
    NUM_PACKETS = 100
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6

    print(f"\n=== Test MH2: {NUM_HOPS} hops, {NUM_PATHS} paths, eps=0.1, "
          f"{NUM_PACKETS} packets ===")

    random.seed(29)  # deterministic
    epsilons = [[0.1] * NUM_PATHS for _ in range(NUM_HOPS)]
    net = MpMhNetwork(
        path_epsilons=epsilons,
        num_packets_to_send=NUM_PACKETS,
        num_paths=NUM_PATHS,
        global_prop_delay=PROP_DELAY,
        num_hops=NUM_HOPS,
        debug=False,
    )
    net.run_sim()
    stats = net.get_simulation_stats()

    expected_packets = set(range(1, NUM_PACKETS + 1))

    decoded_packets = set(net.receiver.information_packets_decoding_times.keys())
    assert decoded_packets == expected_packets, \
        f"Decoded packets mismatch.\n  Missing: {expected_packets - decoded_packets}\n  Extra:   {decoded_packets - expected_packets}"

    # Every decoded packet must be decoded strictly after it was first sent.
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
# MH3 - Verify the chain structure
# ============================================================================

def test_MH3_chain_structure():
    """Build a 3-hop network and verify the wiring:
    SimSender.next_hop -> Node[0] -> Node[1] -> SimReceiver.
    Also verify that after run_sim, every node received and forwarded packets,
    and the receiver decoded the exact expected set."""
    NUM_PACKETS = 50
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6

    print(f"\n=== Test MH3: chain structure verification ===")

    net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, NUM_PACKETS, max_iterations=500)

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
# MH4 - 2 hops (1 node), 4 paths, eps=0
# ============================================================================

def test_MH4_two_hops_one_node():
    """Minimal multi-hop: SimSender -> Node[0] -> SimReceiver."""
    NUM_PACKETS = 100
    NUM_PATHS = 4
    NUM_HOPS = 2
    PROP_DELAY = 6

    print(f"\n=== Test MH4: {NUM_HOPS} hops, {NUM_PATHS} paths, eps=0, "
          f"{NUM_PACKETS} packets ===")

    net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, NUM_PACKETS, max_iterations=1000)

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

    # The single node must have received and forwarded.
    node = net.nodes[0]
    node_new_info = node.my_sender.new_information_packets_history
    assert expected_packets.issubset(node_new_info), \
        f"Node new info buffer should contain all sent info packets.\n  Missing: {expected_packets - node_new_info}"

    print(f"  Decoded:    {decoded_packets == expected_packets}")
    print(f"  Throughput: {net.get_simulation_stats().normalized_throughput:.3f}")
    print(f"  Time:       {net.t} steps")
    print("  PASSED")


# ============================================================================
# MH5 - Timing: lossless per-hop forward arrival
# ============================================================================

def test_MH5_timing_forward_per_hop_lossless():
    """In a lossless network, a packet first sent at t=send_time reaches the
    receiver after hop `h` exactly at send_time + h * hop_prop_delay, and the
    final SimReceiver gets it at send_time + global_prop_delay (= global_rtt // 2)."""
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6  # -> hop_prop_delay=2, hop_rtt=4, global_rtt=12

    print(f"\n=== Test MH5: forward per-hop timing ({NUM_HOPS} hops) ===")

    net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, num_packets=30)

    assert net.hop_prop_delay == PROP_DELAY // NUM_HOPS == 2
    assert net.hop_rtt == net.global_rtt // NUM_HOPS == 4
    assert net.global_rtt == 2 * PROP_DELAY == 12

    # Step long enough for the first packet to reach the receiver.
    for _ in range(net.global_rtt + net.num_hops + 2):
        net.sender.run_step()

    send_time = net.sender.inforamtion_packets_first_transmission_times[1]
    assert send_time == 1, f"First transmission expected at t=1, got {send_time}"

    hpd = net.hop_prop_delay
    for label, hops, receiver in _forward_hops(net):
        expected = send_time + hops * hpd
        arrivals = _first_arrival_times(receiver)
        assert all(a is not None for a in arrivals), \
            f"{label}: some path never received a packet: {arrivals}"
        assert all(a == expected for a in arrivals), \
            f"{label}: expected first arrival at t={expected} " \
            f"(send_time {send_time} + {hops}*hop_prop_delay {hpd}), got {arrivals}"
        print(f"  {label:14s} hops={hops} -> first arrival t={expected} (delta={hops*hpd})")

    # The receiver's arrival delay is one-way = global_prop_delay = global_rtt // 2.
    recv_arrival = _first_arrival_times(net.receiver)[0]
    assert recv_arrival - send_time == net.global_prop_delay == net.num_hops * hpd == net.global_rtt // 2, \
        f"Receiver delay {recv_arrival - send_time} should equal global_prop_delay {net.global_prop_delay} " \
        f"= num_hops*hop_prop_delay = global_rtt//2 ({net.global_rtt // 2})"

    print("  PASSED")


# ============================================================================
# MH6 - Timing: lossless per-hop feedback
# ============================================================================

def test_MH6_timing_feedback_per_hop_lossless():
    """In a lossless network the SimSender hears back (ACK) about its first
    transmission after a full per-hop round trip = hop_rtt:
      - the ACK is created by Node[0] at send_time + hop_prop_delay
      - it arrives back at the SimSender at send_time + hop_rtt
    Also verify every path's feedback channel matches its forward channel's
    per-hop propagation delay."""
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6  # -> hop_prop_delay=2, hop_rtt=4

    print(f"\n=== Test MH6: feedback per-hop timing ({NUM_HOPS} hops) ===")

    net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, num_packets=30)
    hpd = net.hop_prop_delay

    # Feedback channel per-hop delay must equal the forward channel delay.
    for hop_idx, hop_paths in enumerate(net.paths):
        for path in hop_paths:
            assert path.forward_channel.get_propagation_delay() == hpd, \
                f"Hop {hop_idx}: forward delay != hop_prop_delay ({hpd})"
            assert path.feedback_channel.get_propagation_delay() == hpd, \
                f"Hop {hop_idx}: feedback delay != hop_prop_delay ({hpd})"

    # Step tick-by-tick and detect when the SimSender first receives feedback.
    fb_arrival_tick = None
    for _ in range(2 * net.hop_rtt + net.num_hops + 4):
        net.sender.run_step()
        if fb_arrival_tick is None and any(
            len(p.all_feedback_history) > 0 for p in net.sender.paths
        ):
            fb_arrival_tick = net.sender.t

    send_time = net.sender.inforamtion_packets_first_transmission_times[1]
    assert send_time == 1, f"First transmission expected at t=1, got {send_time}"

    assert fb_arrival_tick is not None, "SimSender never received feedback"
    assert fb_arrival_tick - send_time == net.hop_rtt == 2 * hpd, \
        f"SimSender feedback round trip {fb_arrival_tick - send_time} should equal " \
        f"hop_rtt {net.hop_rtt} (= 2*hop_prop_delay {hpd})"

    # The earliest feedback should be an ACK created one hop away (send_time + hpd).
    all_fb = [fb for p in net.sender.paths for fb in p.all_feedback_history]
    earliest = min(all_fb, key=lambda fb: fb.get_creation_time())
    assert earliest.is_ack(), \
        f"Earliest SimSender feedback should be an ACK, got {earliest.get_type()}"
    assert earliest.get_creation_time() - send_time == hpd, \
        f"ACK creation time delta {earliest.get_creation_time() - send_time} " \
        f"should equal one hop delay hop_prop_delay ({hpd})"

    print(f"  ACK created at t={earliest.get_creation_time()} (send_time + {hpd})")
    print(f"  ACK back at SimSender at t={fb_arrival_tick} (send_time + hop_rtt {net.hop_rtt})")
    print("  PASSED")


# ============================================================================
# MH7 - Timing: 2-hop variant
# ============================================================================

def test_MH7_timing_two_hops_lossless():
    """Same forward-timing law with a different topology (2 hops): the receiver
    is reached after num_hops * hop_prop_delay = global_prop_delay = global_rtt // 2."""
    NUM_PATHS = 4
    NUM_HOPS = 2
    PROP_DELAY = 6  # -> hop_prop_delay=3, hop_rtt=6, global_rtt=12

    print(f"\n=== Test MH7: forward timing ({NUM_HOPS} hops) ===")

    net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, num_packets=30)
    assert net.hop_prop_delay == 3 and net.hop_rtt == 6 and net.global_rtt == 12

    for _ in range(net.global_rtt + net.num_hops + 2):
        net.sender.run_step()

    send_time = net.sender.inforamtion_packets_first_transmission_times[1]
    hpd = net.hop_prop_delay

    for label, hops, receiver in _forward_hops(net):
        expected = send_time + hops * hpd
        arrivals = _first_arrival_times(receiver)
        assert all(a == expected for a in arrivals), \
            f"{label}: expected first arrival at t={expected}, got {arrivals}"
        print(f"  {label:14s} hops={hops} -> first arrival t={expected}")

    recv_arrival = _first_arrival_times(net.receiver)[0]
    assert recv_arrival - send_time == net.global_prop_delay == net.global_rtt // 2
    print("  PASSED")


# ============================================================================
# E2E1 - End-to-end feedback round-trip timing
# ============================================================================

def test_E2E1_feedback_roundtrip_timing():
    """In E2E mode, feedback about a packet first transmitted at t0 returns to
    the SimSender exactly at t0 + global_rtt (a full end-to-end round trip),
    and the earliest feedback refers to the packet created at t0."""
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6  # -> hop_prop_delay=2, global_prop_delay=6, global_rtt=12

    print(f"\n=== Test E2E1: end-to-end feedback round-trip timing ===")

    net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, num_packets=30,
                              feedback_source=FeedbackSource.E2E)
    assert net.feedback_source == FeedbackSource.E2E
    assert net.global_rtt == 12 and net.global_prop_delay == 6
    assert set(net.e2e_feedback_channels.keys()) == set(range(1, NUM_PATHS + 1))
    for ch in net.e2e_feedback_channels.values():
        assert ch.get_propagation_delay() == net.global_prop_delay

    fb_arrival_tick = None
    for _ in range(net.global_rtt + net.num_hops + 4):
        net.sender.run_step()
        if fb_arrival_tick is None and any(
            len(p.all_feedback_history) > 0 for p in net.sender.paths
        ):
            fb_arrival_tick = net.sender.t

    send_time = net.sender.inforamtion_packets_first_transmission_times[1]
    assert send_time == 1, f"First transmission expected at t=1, got {send_time}"

    assert fb_arrival_tick is not None, "SimSender never received end-to-end feedback"
    assert fb_arrival_tick - send_time == net.global_rtt, \
        f"E2E feedback round trip {fb_arrival_tick - send_time} should equal " \
        f"global_rtt {net.global_rtt}"

    all_fb = [fb for p in net.sender.paths for fb in p.all_feedback_history]
    earliest = min(all_fb, key=lambda fb: fb.get_related_packet_id().get_creation_time())
    assert earliest.get_related_packet_id().get_creation_time() == send_time, \
        f"Earliest E2E feedback should refer to the packet created at t0={send_time}, " \
        f"got {earliest.get_related_packet_id().get_creation_time()}"

    print(f"  Feedback back at SimSender at t={fb_arrival_tick} (send_time + global_rtt {net.global_rtt})")
    print("  PASSED")


# ============================================================================
# E2E2 - Receiver ACK vs NACK decision (DROPPED marker -> NACK)
# ============================================================================

def test_E2E2_receiver_ack_vs_nack():
    """The SimReceiver in E2E mode emits exactly one end-to-end feedback per
    global path each tick: ACK for every arrival (intermediate nodes recode, so an
    arrival is always a genuine delivery on its carried global path) and NACK for
    any global path that did not arrive (erased on the last hop). Every feedback
    references (carried global path, t - global_prop_delay). Feedback is
    label-complete (one per global path) regardless of which physical path carried
    each label, which is what keeps the source's per-(label, creation_time)
    bookkeeping consistent once nodes re-run natural matching."""
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6

    print(f"\n=== Test E2E2: receiver ACK vs NACK (label-complete) ===")

    net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, num_packets=30,
                              feedback_source=FeedbackSource.E2E)
    receiver = net.receiver
    gpd = net.global_prop_delay
    receiver.t = gpd + 5  # past the warm-up guard so feedback is emitted

    # This tick's arrivals: a NEW carrying global path 1 and a recoded CORRECTION
    # carrying global path 2 (a node filled that slot after an upstream drop).
    # Global paths 3 and 4 did not arrive (erased on the last hop).
    new_pkt = RLNCPacket(global_path_id=1, type=RLNCType.NEW,
                         information_packets=[10, 11],
                         prop_time_left_in_channel=0, creation_time=receiver.t - gpd)
    corr_pkt = RLNCPacket(global_path_id=2, type=NodeRLNCType.CORRECTION,
                          information_packets=[12],
                          prop_time_left_in_channel=0, creation_time=receiver.t - gpd)
    receiver._e2e_arrivals_this_tick = [new_pkt, corr_pkt]
    receiver._send_e2e_feedbacks()

    def _last_on_channel(gp):
        ch = net.e2e_feedback_channels[gp]
        assert len(ch.packets_in_channel) == 1, \
            f"Expected exactly one feedback on e2e channel {gp}, got {len(ch.packets_in_channel)}"
        return ch.packets_in_channel[-1]

    # Both arrivals (NEW on global path 1, recoded CORRECTION on global path 2) -> ACK
    for gp in (1, 2):
        ack = _last_on_channel(gp)
        assert ack.is_ack(), f"Arrival on global path {gp} should yield an ACK, got {ack.get_type()}"
        assert ack.get_related_packet_id().get_creation_time() == receiver.t - gpd
        assert ack.get_related_packet_id().get_global_path_id() == gp

    # Missing global paths 3 and 4 -> NACK each
    for gp in (3, 4):
        nack = _last_on_channel(gp)
        assert nack.is_nack(), f"Missing global path {gp} should yield a NACK, got {nack.get_type()}"
        assert nack.get_related_packet_id().get_creation_time() == receiver.t - gpd
        assert nack.get_related_packet_id().get_global_path_id() == gp

    print("  ACK on every arrival, NACK on missing labels - all correct")
    print("  PASSED")


# ============================================================================
# E2E3 - NodeSender recodes an upstream drop into a CORRECTION (both modes)
# ============================================================================

def test_E2E3_nodesender_recodes_dropped_as_correction():
    """When the matched global-path type is DROPPED (nothing arrived on that global
    path this tick because of an upstream erasure), the NodeSender recodes: it
    re-sends the correction-buffer content as a normal CORRECTION packet, in BOTH
    E2E and HBH modes. Nodes no longer propagate a DROPPED marker end-to-end, so a
    per-hop drop is absorbed rather than forwarded (this keeps each global path at
    its min-cut rate instead of the product of the per-hop erasures)."""
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6

    print(f"\n=== Test E2E3: NodeSender recodes DROPPED -> CORRECTION (E2E and HBH) ===")

    for feedback_source in (FeedbackSource.E2E, FeedbackSource.HBH):
        net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, num_packets=30,
                                  feedback_source=feedback_source)
        node_sender = net.nodes[0].my_sender
        node_sender.correction_information_packets_buffer = {5, 6}
        path = node_sender.paths[0]
        pkt = node_sender.create_rlnc(path, NodeRLNCType.DROPPED)
        assert pkt is not None, \
            f"{feedback_source.name}: a packet should be created when the correction buffer is non-empty"
        assert pkt.get_type() == NodeRLNCType.CORRECTION, \
            f"{feedback_source.name} NodeSender should recode DROPPED into CORRECTION, got {pkt.get_type()}"
        assert set(pkt.get_information_packets()) == {5, 6}, \
            f"{feedback_source.name}: recoded packet must carry the correction-buffer content"

    print("  E2E -> CORRECTION, HBH -> CORRECTION (drop absorbed by recoding)")
    print("  PASSED")


# ============================================================================
# E2E4 - Full decode end-to-end (lossless and light loss)
# ============================================================================

def test_E2E4_full_decode_lossless():
    """With end-to-end feedback and no erasures, every information packet is
    still decoded by the SimReceiver."""
    NUM_PACKETS = 100
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6

    print(f"\n=== Test E2E4: E2E full decode, lossless ({NUM_HOPS} hops) ===")

    net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, NUM_PACKETS,
                              feedback_source=FeedbackSource.E2E)
    net.run_sim()

    expected_packets = set(range(1, NUM_PACKETS + 1))
    decoded_packets = set(net.receiver.information_packets_decoding_times.keys())
    assert decoded_packets == expected_packets, \
        f"Decoded packets mismatch.\n  Missing: {expected_packets - decoded_packets}\n  Extra:   {decoded_packets - expected_packets}"
    assert net.get_simulation_stats().num_transmissions_dropped == 0

    print(f"  Decoded all {NUM_PACKETS} packets in {net.t} steps")
    print("  PASSED")


def test_E2E5_full_decode_light_loss():
    """With end-to-end feedback and light erasures on every hop, the AC-RLNC
    protocol still delivers every packet end-to-end."""
    NUM_PACKETS = 60
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6

    print(f"\n=== Test E2E5: E2E full decode, light loss ({NUM_HOPS} hops) ===")

    random.seed(29)  # deterministic
    epsilons = [[0.1] * NUM_PATHS for _ in range(NUM_HOPS)]
    net = MpMhNetwork(
        path_epsilons=epsilons,
        num_packets_to_send=NUM_PACKETS,
        num_paths=NUM_PATHS,
        global_prop_delay=PROP_DELAY,
        num_hops=NUM_HOPS,
        max_iterations=40000,
        debug=False,
        feedback_source=FeedbackSource.E2E,
    )
    net.run_sim()

    expected_packets = set(range(1, NUM_PACKETS + 1))
    decoded_packets = set(net.receiver.information_packets_decoding_times.keys())
    assert decoded_packets == expected_packets, \
        f"Decoded packets mismatch.\n  Missing: {expected_packets - decoded_packets}\n  Extra:   {decoded_packets - expected_packets}"

    # Sanity: end-to-end feedback actually flowed (both ACKs and NACKs seen).
    assert len(net.sender.acked_feedback_history) > 0, "Expected some E2E ACKs at the sender"
    assert len(net.sender.nacked_feedback_history) > 0, "Expected some E2E NACKs at the sender"

    print(f"  Decoded all {NUM_PACKETS} packets in {net.t} steps")
    print(f"  E2E ACKs: {len(net.sender.acked_feedback_history)}, "
          f"NACKs: {len(net.sender.nacked_feedback_history)}")
    print("  PASSED")


# ============================================================================
# E2E6 - Last node receives per-hop feedback in E2E (for natural matching)
# ============================================================================

def test_E2E6_last_node_receives_per_hop_feedback():
    """In E2E mode the destination SimReceiver still emits per-hop feedback on
    the last hop, so the last Node's sender gets ACK/NACKs on its output paths
    and can estimate r for natural matching (like every other node).

    Lossless: ACKs flow, so every last-node path accumulates feedback and full
    decode still succeeds.
    Light loss: NACKs move at least one last-node path's r off its initial value
    (1 - initial_epsilon = 1.0)."""
    NUM_PATHS = 4
    NUM_HOPS = 3
    PROP_DELAY = 6

    print(f"\n=== Test E2E6: last node per-hop feedback (E2E) ===")

    # --- Lossless: feedback flows to the last node, full decode still works ---
    net = _build_lossless_net(NUM_HOPS, NUM_PATHS, PROP_DELAY, num_packets=100,
                              feedback_source=FeedbackSource.E2E)
    net.run_sim()

    last_node_sender = net.nodes[-1].my_sender
    for path in last_node_sender.paths:
        assert len(path.all_feedback_history) > 0, \
            "Last node path received no per-hop feedback in E2E (lossless)"

    expected_packets = set(range(1, 100 + 1))
    decoded_packets = set(net.receiver.information_packets_decoding_times.keys())
    assert decoded_packets == expected_packets, \
        f"Decoded packets mismatch.\n  Missing: {expected_packets - decoded_packets}\n  Extra:   {decoded_packets - expected_packets}"

    # --- Light loss: at least one last-node path adapts r away from initial ---
    random.seed(29)  # deterministic
    NUM_PACKETS = 60
    epsilons = [[0.1] * NUM_PATHS for _ in range(NUM_HOPS)]
    net_loss = MpMhNetwork(
        path_epsilons=epsilons,
        num_packets_to_send=NUM_PACKETS,
        num_paths=NUM_PATHS,
        global_prop_delay=PROP_DELAY,
        num_hops=NUM_HOPS,
        max_iterations=40000,
        debug=False,
        feedback_source=FeedbackSource.E2E,
    )
    net_loss.run_sim()

    last_node_sender = net_loss.nodes[-1].my_sender
    initial_r = 1.0 - last_node_sender.initial_epsilon
    for path in last_node_sender.paths:
        assert len(path.all_feedback_history) > 0, \
            "Last node path received no per-hop feedback in E2E (light loss)"
    adapted = [p for p in last_node_sender.paths if p.r != initial_r]
    assert len(adapted) > 0, \
        f"No last-node path adapted r off its initial value {initial_r} under loss"

    print(f"  Last node paths with feedback: {len(last_node_sender.paths)}, "
          f"adapted r: {len(adapted)}")
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
    test_MH5_timing_forward_per_hop_lossless()
    test_MH6_timing_feedback_per_hop_lossless()
    test_MH7_timing_two_hops_lossless()
    test_E2E1_feedback_roundtrip_timing()
    test_E2E2_receiver_ack_vs_nack()
    test_E2E3_nodesender_recodes_dropped_as_correction()
    test_E2E4_full_decode_lossless()
    test_E2E5_full_decode_light_loss()
    test_E2E6_last_node_receives_per_hop_feedback()

    print("\n" + "=" * 70)
    print("ALL MP-MH NETWORK TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
