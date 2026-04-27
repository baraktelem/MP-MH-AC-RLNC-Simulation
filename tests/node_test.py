"""
Tests for Node class (NodeReceiver + NodeSender working together).

Packets are injected directly into the NodeReceiver's input-path
arrived_packets buffer.  After node.run_step() the NodeSender should
have created and sent the right packets on the output paths.

Test naming convention:
- N1:   1 path, 1 NEW packet passes through the Node
- N2:   1 path, each correction type (FEC / FB_FEC / CORRECTION) passes through
- N3:   4 paths, mixed NEW and correction types all pass through
- N4:   3 input-path epsilons [0, 0.3, 0.8] — delivery ordering (SimSender → Node)
- N5:   Output packet info equals accumulated receiver buffer
- N6:   Buffers accumulate correctly across multiple steps
- N7:   Natural matching sorts sender output-paths by r
- N8:   Feedback (ACK) is placed on the input-path feedback channel
"""

import sys
import os
import random
from contextlib import redirect_stdout
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Channels import Path
from Packet import Packet, RLNCPacket, RLNCType, NodeRLNCType
from Node import Node
from Sender import SimSender
from Receiver import SimReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockNetwork:
    """Minimal stand-in for MpMhNetwork — only provides global_paths_idx_by_r."""
    def __init__(self, num_paths: int):
        self.global_paths_idx_by_r = list(range(1, num_paths + 1))


def make_rlnc(
    info: list[int],
    global_path_id: int,
    rlnc_type: RLNCType = RLNCType.NEW,
    creation_time: int = 0,
) -> RLNCPacket:
    """Packet with prop_time=0 ready for direct injection into arrived_packets."""
    return RLNCPacket(
        global_path_id=global_path_id,
        type=rlnc_type,
        information_packets=info,
        prop_time_left_in_channel=0,
        creation_time=creation_time,
    )


def create_paths_with_epsilons(
    epsilons: list[float],
    prop_delay: int = 2,
    hop_index: int = 0,
) -> list[Path]:
    """Create paths with 1-based global indices (matching MpMhNetwork)."""
    paths = []
    for i, eps in enumerate(epsilons):
        p = Path(prop_delay, eps, hop_index, i)
        p.set_global_path_index(i + 1)
        paths.append(p)
    return paths


def create_node_setup(
    num_paths: int,
    prop_delay: int = 2,
    input_epsilons: list[float] | None = None,
    output_epsilons: list[float] | None = None,
):
    """Build a Node with input/output paths and a MockNetwork."""
    if input_epsilons is None:
        input_epsilons = [0.0] * num_paths
    if output_epsilons is None:
        output_epsilons = [0.0] * num_paths

    input_paths = create_paths_with_epsilons(input_epsilons, prop_delay, hop_index=0)
    output_paths = create_paths_with_epsilons(output_epsilons, prop_delay, hop_index=1)
    mock = MockNetwork(num_paths)
    rtt = prop_delay * 2
    node = Node(
        hop_num=0,
        input_paths=input_paths,
        output_paths=output_paths,
        rtt=rtt,
        Network=mock,
    )
    return node, input_paths, output_paths, mock


def inject_to_input(node: Node, path_index: int, packet: RLNCPacket):
    """Place a packet directly in the NodeReceiver's arrived buffer."""
    node.my_receiver.receiver_paths[path_index].forward_channel.arrived_packets.append(packet)


def get_output_channel_history(output_path: Path) -> list[RLNCPacket]:
    return output_path.forward_channel.get_channel_history()


# ============================================================================
# N1 — 1 path, 1 NEW packet passes through the Node
# ============================================================================

def test_N1_single_path_new_packet_passes():
    """Inject a NEW packet on the single input path.
    Verify the NodeSender creates and sends a NEW packet on the output path
    with the same information packets."""
    print("\n=== Test N1: 1 path, 1 NEW packet passes ===")
    node, _, output_paths, _ = create_node_setup(num_paths=1)

    pkt = make_rlnc([1, 2, 3], global_path_id=1)
    inject_to_input(node, 0, pkt)

    with redirect_stdout(open(os.devnull, "w")):
        node.run_step(time=1)

    # --- receiver ---
    assert len(node.my_receiver.new_rlnc_packets_history) == 1
    assert node.my_receiver.new_information_packets_buffer == {1, 2, 3}

    # --- sender ---
    assert len(node.my_sender.new_rlnc_packets_history) == 1
    sent = node.my_sender.new_rlnc_packets_history[0]
    assert sent.get_type() == RLNCType.NEW
    assert set(sent.get_information_packets()) == {1, 2, 3}

    # --- output channel ---
    out = get_output_channel_history(output_paths[0])
    assert len(out) == 1
    assert out[0].get_type() == RLNCType.NEW
    assert set(out[0].get_information_packets()) == {1, 2, 3}

    print("  PASSED")


# ============================================================================
# N2 — 1 path, each correction type passes through
# ============================================================================

def test_N2_correction_types_pass_through():
    """For each of FEC, FB_FEC, CORRECTION: inject on 1 input path, verify
    the NodeSender always sends a CORRECTION packet from the correction buffer
    (Nodes convert every non-NEW input into CORRECTION on re-transmission)."""
    print("\n=== Test N2: 1 path, FEC / FB_FEC / CORRECTION → CORRECTION ===")

    for rlnc_type in [RLNCType.FEC, RLNCType.FB_FEC, NodeRLNCType.CORRECTION]:
        node, _, output_paths, _ = create_node_setup(num_paths=1)

        pkt = make_rlnc([10, 20], global_path_id=1, rlnc_type=rlnc_type)
        inject_to_input(node, 0, pkt)

        with redirect_stdout(open(os.devnull, "w")):
            node.run_step(time=1)

        # receiver: goes to correction buffer, not new
        assert node.my_receiver.correction_information_packets_buffer == {10, 20}
        assert node.my_receiver.new_information_packets_buffer == set()

        # sender: always outputs CORRECTION regardless of input type
        assert len(node.my_sender.correction_packets_history) == 1
        sent = node.my_sender.correction_packets_history[0]
        assert sent.get_type() == NodeRLNCType.CORRECTION, \
            f"Input {rlnc_type.name} should produce CORRECTION, got {sent.get_type().name}"
        assert set(sent.get_information_packets()) == {10, 20}

        # output channel
        out = get_output_channel_history(output_paths[0])
        assert len(out) == 1
        assert out[0].get_type() == NodeRLNCType.CORRECTION, \
            f"Output for {rlnc_type.name} input should be CORRECTION, got {out[0].get_type().name}"

        print(f"  {rlnc_type.name} → CORRECTION: PASSED")

    print("  ALL PASSED")


# ============================================================================
# N3 — 4 paths, mixed NEW and correction types
# ============================================================================

def test_N3_four_paths_mixed_types():
    """Inject NEW on paths 0-1 and FEC / CORRECTION on paths 2-3.
    Verify each output path gets the matching type and the correct info buffer."""
    print("\n=== Test N3: 4 paths, mixed NEW + correction ===")
    node, _, output_paths, _ = create_node_setup(num_paths=4)

    inject_to_input(node, 0, make_rlnc([1, 2], global_path_id=1, rlnc_type=RLNCType.NEW))
    inject_to_input(node, 1, make_rlnc([3, 4], global_path_id=2, rlnc_type=RLNCType.NEW))
    inject_to_input(node, 2, make_rlnc([5, 6], global_path_id=3, rlnc_type=RLNCType.FEC))
    inject_to_input(node, 3, make_rlnc([7, 8], global_path_id=4, rlnc_type=NodeRLNCType.CORRECTION))

    with redirect_stdout(open(os.devnull, "w")):
        node.run_step(time=1)

    # --- receiver buffers ---
    assert node.my_receiver.new_information_packets_buffer == {1, 2, 3, 4}
    assert node.my_receiver.correction_information_packets_buffer == {5, 6, 7, 8}

    # --- sender totals ---
    assert len(node.my_sender.new_rlnc_packets_history) == 2
    assert len(node.my_sender.correction_packets_history) == 2

    # --- per-output-path checks ---
    # All sender paths have r=1.0, so natural matching preserves the order:
    #   sender.paths[0] → global 1 → NEW,       sender.paths[1] → global 2 → NEW,
    #   sender.paths[2] → global 3 → CORRECTION, sender.paths[3] → global 4 → CORRECTION
    # (FEC / CORRECTION inputs are both re-transmitted as CORRECTION by the Node)
    expected_types = [RLNCType.NEW, RLNCType.NEW, NodeRLNCType.CORRECTION, NodeRLNCType.CORRECTION]
    expected_info = [
        {1, 2, 3, 4},  # NEW → new buffer
        {1, 2, 3, 4},
        {5, 6, 7, 8},  # correction → correction buffer
        {5, 6, 7, 8},
    ]
    for i in range(4):
        out = get_output_channel_history(output_paths[i])
        assert len(out) == 1, \
            f"Output path {i}: expected 1 packet, got {len(out)}"
        assert out[0].get_type() == expected_types[i], \
            f"Output path {i}: expected {expected_types[i].name}, got {out[0].get_type().name}"
        assert set(out[0].get_information_packets()) == expected_info[i], \
            f"Output path {i}: info mismatch"

    print("  PASSED")


# ============================================================================
# N4 — Natural Matching: NodeSender syncs its path ordering with the network
# ============================================================================

def test_N4_natural_matching_syncs_with_network():
    """Natural Matching: the NodeSender must assign its local output paths
    so that the path with the highest r gets the best global path index
    (as dictated by the network's global_paths_idx_by_r).

    Setup: 3 output paths with different r values.  The network publishes
    a global ordering [3, 1, 2] (meaning global path 3 is best, then 1,
    then 2).  After perform_natural_matching the NodeSender's paths should
    be re-assigned:
        local best  (r=0.9) → global 3
        local mid   (r=0.5) → global 1
        local worst (r=0.2) → global 2
    """
    print("\n=== Test N4: Natural Matching — NodeSender syncs with network ===")

    NUM_PATHS = 3
    node, _, output_paths, mock = create_node_setup(num_paths=NUM_PATHS)
    sender = node.my_sender

    # Manually set different r values on the 3 sender paths
    sender.paths[0].r = 0.5   # mid
    sender.paths[1].r = 0.2   # worst
    sender.paths[2].r = 0.9   # best

    # Network publishes a non-trivial global ordering: best=3, mid=1, worst=2
    mock.global_paths_idx_by_r = [3, 1, 2]

    sender.perform_natural_matching()

    # After matching, local paths sorted by r descending [path2, path0, path1]
    # are zipped with global order [3, 1, 2]:
    #   path2 (r=0.9, best)  → global 3
    #   path0 (r=0.5, mid)   → global 1
    #   path1 (r=0.2, worst) → global 2
    assert sender.paths[2].get_global_path_index() == 3, \
        f"Best local path (r=0.9) should map to global 3, got {sender.paths[2].get_global_path_index()}"
    assert sender.paths[0].get_global_path_index() == 1, \
        f"Mid local path (r=0.5) should map to global 1, got {sender.paths[0].get_global_path_index()}"
    assert sender.paths[1].get_global_path_index() == 2, \
        f"Worst local path (r=0.2) should map to global 2, got {sender.paths[1].get_global_path_index()}"

    print(f"  global_paths_idx_by_r = {mock.global_paths_idx_by_r}")
    print(f"  path0 (r=0.5) → global {sender.paths[0].get_global_path_index()}")
    print(f"  path1 (r=0.2) → global {sender.paths[1].get_global_path_index()}")
    print(f"  path2 (r=0.9) → global {sender.paths[2].get_global_path_index()}")
    print("  PASSED")


# ============================================================================
# N5 — Output info matches accumulated receiver buffer
# ============================================================================

def test_N5_output_info_matches_accumulated_buffer():
    """Inject NEW packets over 2 steps so the buffer accumulates.
    The last sender output must carry the full accumulated buffer."""
    print("\n=== Test N5: Output info = accumulated receiver buffer ===")
    node, _, output_paths, _ = create_node_setup(num_paths=1)

    inject_to_input(node, 0, make_rlnc([1, 2], global_path_id=1, creation_time=0))
    with redirect_stdout(open(os.devnull, "w")):
        node.run_step(time=1)

    inject_to_input(node, 0, make_rlnc([2, 3, 4], global_path_id=1, creation_time=1))
    with redirect_stdout(open(os.devnull, "w")):
        node.run_step(time=2)

    assert node.my_receiver.new_information_packets_buffer == {1, 2, 3, 4}

    last_sent = node.my_sender.new_rlnc_packets_history[-1]
    assert set(last_sent.get_information_packets()) == {1, 2, 3, 4}, \
        f"Expected {{1,2,3,4}}, got {set(last_sent.get_information_packets())}"

    print("  PASSED")


# ============================================================================
# N6 — Buffers accumulate correctly across multiple steps
# ============================================================================

def test_N6_buffers_accumulate_across_steps():
    """Two steps, 2 paths each step: NEW on path 0, FEC on path 1.
    Both buffers should grow as set-unions."""
    print("\n=== Test N6: Buffers accumulate across steps ===")
    node, _, output_paths, _ = create_node_setup(num_paths=2)

    # --- step 1 ---
    inject_to_input(node, 0, make_rlnc([1], global_path_id=1, rlnc_type=RLNCType.NEW, creation_time=0))
    inject_to_input(node, 1, make_rlnc([10], global_path_id=2, rlnc_type=RLNCType.FEC, creation_time=0))
    with redirect_stdout(open(os.devnull, "w")):
        node.run_step(time=1)

    assert node.my_receiver.new_information_packets_buffer == {1}
    assert node.my_receiver.correction_information_packets_buffer == {10}

    # --- step 2 ---
    inject_to_input(node, 0, make_rlnc([2, 3], global_path_id=1, rlnc_type=RLNCType.NEW, creation_time=1))
    inject_to_input(node, 1, make_rlnc([11, 12], global_path_id=2, rlnc_type=RLNCType.FEC, creation_time=1))
    with redirect_stdout(open(os.devnull, "w")):
        node.run_step(time=2)

    assert node.my_receiver.new_information_packets_buffer == {1, 2, 3}
    assert node.my_receiver.correction_information_packets_buffer == {10, 11, 12}

    # sender produced 2 NEW + 2 correction (one per path per step)
    assert len(node.my_sender.new_rlnc_packets_history) == 2
    assert len(node.my_sender.correction_packets_history) == 2

    print("  PASSED")


# ============================================================================
# N7 — Natural matching sorts sender output-paths by r
# ============================================================================

def test_N7_natural_matching_sorts_by_r():
    """Manually set r values on the 3 sender paths and verify
    perform_natural_matching assigns the best global index to the
    best local path."""
    print("\n=== Test N7: Natural matching sorts by r ===")
    node, _, _, mock = create_node_setup(num_paths=3)
    sender = node.my_sender

    # Manually override r: path[0]=worst, path[1]=mid, path[2]=best
    sender.paths[0].r = 0.3
    sender.paths[1].r = 0.7
    sender.paths[2].r = 0.9

    # Global ordering (best first): [1, 2, 3]
    mock.global_paths_idx_by_r = [1, 2, 3]

    sender.perform_natural_matching()

    # best local (r=0.9) → best global (1)
    assert sender.paths[2].get_global_path_index() == 1, \
        f"Expected global 1 for best path, got {sender.paths[2].get_global_path_index()}"
    # middle local (r=0.7) → middle global (2)
    assert sender.paths[1].get_global_path_index() == 2, \
        f"Expected global 2 for mid path, got {sender.paths[1].get_global_path_index()}"
    # worst local (r=0.3) → worst global (3)
    assert sender.paths[0].get_global_path_index() == 3, \
        f"Expected global 3 for worst path, got {sender.paths[0].get_global_path_index()}"

    print("  PASSED")


# ============================================================================
# N8 — Feedback (ACK) is placed on the input-path feedback channel
# ============================================================================

def test_N8_feedback_on_input_paths():
    """Inject a packet on one input path and run only the receiver.
    Verify an ACK appears in that path's feedback channel history."""
    print("\n=== Test N8: Feedback sent on input paths ===")
    node, _, _, _ = create_node_setup(num_paths=2)

    pkt = make_rlnc([1], global_path_id=1, creation_time=0)
    inject_to_input(node, 0, pkt)

    # Run only the receiver (the sender would assert because path 1 has no type)
    with redirect_stdout(open(os.devnull, "w")):
        node.my_receiver.run_step(time=1)

    # Path 0: should have 1 ACK
    fb_0 = node.my_receiver.get_receiver_path(0).get_sent_feedback_channel_history()
    assert len(fb_0) == 1, f"Expected 1 feedback on path 0, got {len(fb_0)}"
    assert fb_0[0].is_ack(), "Feedback on path 0 should be ACK"

    # Path 1: no packet, and t <= prop_delay so no NACK either
    fb_1 = node.my_receiver.get_receiver_path(1).get_sent_feedback_channel_history()
    assert len(fb_1) == 0, f"Expected 0 feedback on path 1, got {len(fb_1)}"

    print("  PASSED")


# ============================================================================
# N9 — SimSender → Node
# ============================================================================

def test_N9_sim_sender_to_node():
    """SimSender sends on hop-0 paths, NodeReceiver receives after propagation,
    NodeSender re-transmits on hop-1 paths.  Run 20 steps, verify the full
    pipeline works without crashes and packets flow through."""
    print("\n=== Test N9: SimSender → Node (20 steps) ===")

    NUM_PATHS = 2
    PROP_DELAY = 2
    RTT = PROP_DELAY * 2
    NUM_STEPS = 20

    random.seed(42)

    paths_hop0 = create_paths_with_epsilons([0.0] * NUM_PATHS, PROP_DELAY, hop_index=0)
    paths_hop1 = create_paths_with_epsilons([0.0] * NUM_PATHS, PROP_DELAY, hop_index=1)
    mock = MockNetwork(NUM_PATHS)
    node = Node(0, paths_hop0, paths_hop1, RTT, Network=mock)

    sender = SimSender(
        num_of_packets_to_send=10_000,
        rtt=RTT,
        paths=paths_hop0,
        initial_epsilon=0.0,
        next_hop=node,
        network=mock,
    )

    with redirect_stdout(open(os.devnull, "w")):
        for t in range(1, NUM_STEPS + 1):
            sender.run_step()
            if t == NUM_STEPS - PROP_DELAY:
                # Get supposed buffer states of the NodeSender
                new_rlnc_packets_sent = [rlnc for rlnc in sender.sent_new_rlnc_history]
                new_information_packets_sent = set([p for rlnc in new_rlnc_packets_sent for p in rlnc.get_information_packets()])
                sent_correction_rlnc_history = sender.sent_fec_history + sender.sent_fb_fec_history
                correction_information_packets_sent = set([p for rlnc in sent_correction_rlnc_history for p in rlnc.get_information_packets()])

                # Validate SimSender history collection
                new_rlnc_val = all(rlnc.get_type() == RLNCType.NEW for rlnc in new_rlnc_packets_sent)
                assert new_rlnc_val, "All new packets in SimSender.sent_new_rlnc_history should be of type NEW"
                fec_rlnc_val = all(fec.get_type() == RLNCType.FEC for fec in sender.sent_fec_history)
                assert fec_rlnc_val, "All FEC packets in SimSender.sent_fec_history should be of type FEC"
                fb_fec_rlnc_val = all(fb_fec.get_type() == RLNCType.FB_FEC for fb_fec in sender.sent_fb_fec_history)
                assert fb_fec_rlnc_val, "All FB-FEC packets in SimSender.sent_fb_fec_history should be of type FB-FEC"

    # NodeReceiver received packets from SimSender
    nr_received = len(node.my_receiver.get_received_rlnc_channel_history())
    assert nr_received == NUM_PATHS * (NUM_STEPS - PROP_DELAY), f"NodeReceiver should have received {NUM_STEPS* (NUM_PATHS - PROP_DELAY)} packets, got {nr_received}"

    # NodeSender forwarded packets onto hop1
    ns_new = len(node.my_sender.new_rlnc_packets_history)
    ns_corr = len(node.my_sender.correction_packets_history)
    assert ns_new + ns_corr == NUM_PATHS * (NUM_STEPS - PROP_DELAY), f"NodeSender should have sent {NUM_PATHS * (NUM_STEPS - PROP_DELAY)} packets, got {ns_new + ns_corr}"

    assert node.my_sender.new_information_packets_history == new_information_packets_sent, f"NodeSender new information packets history should be:\n{new_information_packets_sent},\n got: \n{node.my_sender.new_information_packets_history}"
    assert node.my_sender.correction_information_packets_history == correction_information_packets_sent, f"NodeSender correction information packets history should be:\n{correction_information_packets_sent},\n got: \n{node.my_sender.correction_information_packets_history}"

    # # hop1 forward channels contain the forwarded packets
    # for i, path in enumerate(paths_hop1):
    #     h = path.forward_channel.get_channel_history()
    #     assert len(h) == NUM_STEPS - PROP_DELAY, f"hop1 path {i} should have {NUM_STEPS - PROP_DELAY} packets in channel history, got {len(h)}"

    # SimSender received feedback from the NodeReceiver
    sender_fb = len(sender.acked_feedback_history) + len(sender.nacked_feedback_history)
    num_feedback_to_receive = NUM_PATHS * (NUM_STEPS - 2*PROP_DELAY) # 2*PROP_DELAY because no fb is sent before:
    # 1. Getting the first packet (1 PROP_DELAY), 
    # 2. No fb that was sent at the last PROP_DELAY steps will reach the SimSender
    assert sender_fb == num_feedback_to_receive, f"SimSender should have received {num_feedback_to_receive} feedbacks, got {sender_fb}"

    

    print(f"  NodeReceiver received:  {nr_received}")
    print(f"  NodeSender sent:        {ns_new} NEW + {ns_corr} CORR")
    print(f"  SimSender got feedback: {sender_fb}")
    print("  PASSED")


# ============================================================================
# N10 — Node → Node
# ============================================================================

def test_N10_node_to_node():
    """Two Nodes chained: node1.next_hop = node2.
    node1.run_step() cascades into node2.run_step().
    Node2's sender safely skips during the first PROP_DELAY steps
    (before any packets arrive) thanks to the early-return guard."""
    NUM_PACKETS_TO_SEND = 100
    print(f"\n=== Test N10: Node → Node with next_hop chaining ({NUM_PACKETS_TO_SEND} packets) ===")

    NUM_PATHS = 2
    PROP_DELAY = 2
    RTT = PROP_DELAY * 2

    packet_types_map : dict[int, NodeRLNCType | RLNCType] = {
        0: RLNCType.NEW,
        1: RLNCType.FEC,
        2: RLNCType.FB_FEC,
        3: NodeRLNCType.CORRECTION,
        4: NodeRLNCType.DROPPED,
    }

    @dataclass
    class PacketTracker:
        injected : list[Packet] = field(default_factory=list)
        expected_received : list[Packet] = field(default_factory=list)

    packets_injected_map : dict[NodeRLNCType | RLNCType, PacketTracker] = {
        RLNCType.NEW: PacketTracker(),
        RLNCType.FEC: PacketTracker(),
        RLNCType.FB_FEC: PacketTracker(),
        NodeRLNCType.CORRECTION: PacketTracker(),
        NodeRLNCType.DROPPED: 0,
    }

    paths_hop0 = create_paths_with_epsilons([0.0] * NUM_PATHS, PROP_DELAY, hop_index=0)
    paths_hop1 = create_paths_with_epsilons([0.0] * NUM_PATHS, PROP_DELAY, hop_index=1)
    paths_hop2 = create_paths_with_epsilons([0.0] * NUM_PATHS, PROP_DELAY, hop_index=2)
    mock = MockNetwork(NUM_PATHS)

    node2 = Node(1, paths_hop1, paths_hop2, RTT, Network=mock)
    node1 = Node(0, paths_hop0, paths_hop1, RTT, next_hop=node2, Network=mock)
    
    information_packets_injected : list[int] = []
    new_information = 0
    num_steps = (NUM_PACKETS_TO_SEND // NUM_PATHS)
    with redirect_stdout(open(os.devnull, "w")):
        for t in range(1, num_steps + 1):
            for i in range(NUM_PATHS):
                # Select packet type
                type = packet_types_map[t % len(packet_types_map)] if t > 1 else RLNCType.NEW
                if type == NodeRLNCType.DROPPED: # Drop the packet
                    packets_injected_map[NodeRLNCType.DROPPED] += 1
                    continue
                if type == RLNCType.NEW: # Add new information packet
                    new_information += 1
                    information_packets_injected.append(new_information)
                pkt = make_rlnc(info=information_packets_injected, global_path_id=i + 1, creation_time=t, rlnc_type=type)
                inject_to_input(
                    node1, i,
                    pkt,
                )
                # Track number of packets injected and received
                pkt_tracker = packets_injected_map[type]
                pkt_tracker.injected.append(pkt)
                if t > PROP_DELAY and t < num_steps - PROP_DELAY:
                    pkt_tracker.expected_received.append(pkt)
            node1.run_step(time=t)

    n1_received = len(node1.my_receiver.get_received_rlnc_channel_history())
    n2_received = len(node2.my_receiver.get_received_rlnc_channel_history())

    num_dropped = packets_injected_map[NodeRLNCType.DROPPED]
    assert n1_received == NUM_PACKETS_TO_SEND - num_dropped, \
        f"Node-1 should have received {NUM_PACKETS_TO_SEND - num_dropped} packets, got: {n1_received}"
    assert n2_received == NUM_PACKETS_TO_SEND - (PROP_DELAY * NUM_PATHS), \
        f"Node-2 should have received {NUM_PACKETS_TO_SEND - num_dropped - (PROP_DELAY * NUM_PATHS)} packets, got: {n2_received}"

    n1_new_information = node1.my_sender.new_information_packets_history
    new_information_rlnc_injected = packets_injected_map[RLNCType.NEW].injected
    new_information_packets_injected_set = {pkt for p in new_information_rlnc_injected for pkt in p.get_information_packets()}
    assert n1_new_information == new_information_packets_injected_set, \
        f"Node-1 should have the same new information packets as the injected packets,\
            \ngot:\n{n1_new_information}\n\expected:\n{new_information_packets_injected_set}"

    n1_correction_information = node1.my_sender.correction_information_packets_history
    correction_rlnc_injected = packets_injected_map[RLNCType.FEC].injected + \
        packets_injected_map[RLNCType.FB_FEC].injected + \
        packets_injected_map[NodeRLNCType.CORRECTION].injected
    correction_packets_injected_set = {pkt for p in correction_rlnc_injected for pkt in p.get_information_packets()}
    assert n1_correction_information == correction_packets_injected_set, \
        f"Node-1 should have the same correction information packets as the injected packets,\
            \ngot:\n{n1_correction_information}\nexpected:\n{correction_packets_injected_set}"

    n2_sent_new_information = node2.my_sender.new_information_packets_history
    new_information_rlnc_expected = packets_injected_map[RLNCType.NEW].expected_received
    new_information_packets_expected_set = {pkt for p in new_information_rlnc_expected for pkt in p.get_information_packets()}
    assert n2_sent_new_information == new_information_packets_expected_set, \
        f"Node-2 should have the same new information packets as the injected packets,\
            \ngot:\n{n2_sent_new_information}\nexpected:\n{new_information_packets_expected_set}"

    n2_sent_correction_information = node2.my_sender.correction_information_packets_history
    correction_rlnc_expected = packets_injected_map[RLNCType.FEC].expected_received + \
        packets_injected_map[RLNCType.FB_FEC].expected_received + \
        packets_injected_map[NodeRLNCType.CORRECTION].expected_received
    correction_packets_expected_set = {pkt for p in correction_rlnc_expected for pkt in p.get_information_packets()}
    assert n2_sent_correction_information == correction_packets_expected_set, \
        f"Node-2 should have the same correction information packets as the injected packets,\
            \ngot:\n{n2_sent_correction_information}\nexpected:\n{correction_packets_expected_set}"
    
    print(f"  Node-1 received: {n1_received}")
    print(f"  Node-2 received: {n2_received}")
    print(f"  Num packets injected: {NUM_PACKETS_TO_SEND}")
    print("  PASSED")



# ============================================================================
# N11 — Node → SimReceiver
# ============================================================================

def test_N11_node_to_sim_receiver():
    """Node with next_hop=SimReceiver.  Inject mixed packet types (NEW,
    FEC, FB_FEC, CORRECTION, DROPPED) on the Node's input.  The cascade
    node.run_step() → receiver + sender + sim_receiver.run_step() runs
    the full pipeline.  Verify the SimReceiver receives and decodes."""
    NUM_PACKETS_TO_SEND = 100
    print(f"\n=== Test N11: Node → SimReceiver with next_hop chaining ({NUM_PACKETS_TO_SEND} packets) ===")

    NUM_PATHS = 2
    PROP_DELAY = 2
    RTT = PROP_DELAY * 2

    packet_types_map: dict[int, NodeRLNCType | RLNCType] = {
        0: RLNCType.NEW,
        1: RLNCType.FEC,
        2: RLNCType.FB_FEC,
        3: NodeRLNCType.CORRECTION,
        4: NodeRLNCType.DROPPED,
    }

    @dataclass
    class PacketTracker:
        injected: list[Packet] = field(default_factory=list)
        expected_received: list[Packet] = field(default_factory=list)

    packets_injected_map: dict[NodeRLNCType | RLNCType, PacketTracker] = {
        RLNCType.NEW: PacketTracker(),
        RLNCType.FEC: PacketTracker(),
        RLNCType.FB_FEC: PacketTracker(),
        NodeRLNCType.CORRECTION: PacketTracker(),
        NodeRLNCType.DROPPED: 0,
    }

    paths_hop0 = create_paths_with_epsilons([0.0] * NUM_PATHS, PROP_DELAY, hop_index=0)
    paths_hop1 = create_paths_with_epsilons([0.0] * NUM_PATHS, PROP_DELAY, hop_index=1)
    mock = MockNetwork(NUM_PATHS)

    sim_receiver = SimReceiver(paths_hop1, RTT)
    node = Node(0, paths_hop0, paths_hop1, RTT, next_hop=sim_receiver, Network=mock)

    information_packets_injected: list[int] = []
    new_information = 0
    num_steps = NUM_PACKETS_TO_SEND // NUM_PATHS
    with redirect_stdout(open(os.devnull, "w")):
        for t in range(1, num_steps + 1):
            for i in range(NUM_PATHS):
                pkt_type = packet_types_map[t % len(packet_types_map)] if t > 1 else RLNCType.NEW
                if pkt_type == NodeRLNCType.DROPPED:
                    packets_injected_map[NodeRLNCType.DROPPED] += 1
                    continue
                if pkt_type == RLNCType.NEW:
                    new_information += 1
                    information_packets_injected.append(new_information)
                pkt = make_rlnc(
                    info=information_packets_injected,
                    global_path_id=i + 1,
                    creation_time=t,
                    rlnc_type=pkt_type,
                )
                inject_to_input(node, i, pkt)
                pkt_tracker = packets_injected_map[pkt_type]
                pkt_tracker.injected.append(pkt)
                if t > PROP_DELAY and t < num_steps - PROP_DELAY:
                    pkt_tracker.expected_received.append(pkt)
            node.run_step(time=t)

    # --- Node receiver ---
    n_received = len(node.my_receiver.get_received_rlnc_channel_history())
    total_injected = sum(len(pt.injected) for pt in packets_injected_map.values() if isinstance(pt, PacketTracker))
    assert n_received == total_injected, \
        f"Node should have received {total_injected} packets, got: {n_received}"

    # --- Node sender info buffers ---
    n_new_info = node.my_sender.new_information_packets_history
    new_rlnc_injected = packets_injected_map[RLNCType.NEW].injected
    new_info_expected = {p for rlnc in new_rlnc_injected for p in rlnc.get_information_packets()}
    assert n_new_info == new_info_expected, \
        f"Node new info mismatch,\ngot:\n{n_new_info}\nexpected:\n{new_info_expected}"

    n_corr_info = node.my_sender.correction_information_packets_history
    corr_rlnc_injected = (
        packets_injected_map[RLNCType.FEC].injected +
        packets_injected_map[RLNCType.FB_FEC].injected +
        packets_injected_map[NodeRLNCType.CORRECTION].injected
    )
    corr_info_expected = {p for rlnc in corr_rlnc_injected for p in rlnc.get_information_packets()}
    assert n_corr_info == corr_info_expected, \
        f"Node correction info mismatch,\ngot:\n{n_corr_info}\nexpected:\n{corr_info_expected}"

    # --- SimReceiver ---
    sr_received = len(sim_receiver.get_received_rlnc_channel_history())
    sr_decoded = len(sim_receiver.information_packets_decoding_times)
    assert sr_received > 0, "SimReceiver should have received packets"
    assert sr_decoded > 0, "SimReceiver should have decoded some info packets"

    # --- Feedback ---
    node_sender_fb = sum(len(p.all_feedback_history) for p in node.my_sender.paths)
    assert node_sender_fb > 0, "Node sender should have received feedback from SimReceiver"

    print(f"  Node received:        {n_received}")
    print(f"  SimReceiver received: {sr_received}")
    print(f"  SimReceiver decoded:  {sr_decoded} info packets")
    print(f"  Node sender feedback: {node_sender_fb}")
    print(f"  Num packets injected: {NUM_PACKETS_TO_SEND}")
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING NODE TESTS")
    print("=" * 70)

    test_N1_single_path_new_packet_passes()
    test_N2_correction_types_pass_through()
    test_N3_four_paths_mixed_types()
    test_N4_natural_matching_syncs_with_network()
    test_N5_output_info_matches_accumulated_buffer()
    test_N6_buffers_accumulate_across_steps()
    test_N7_natural_matching_sorts_by_r()
    test_N8_feedback_on_input_paths()
    test_N9_sim_sender_to_node()
    test_N10_node_to_node()
    test_N11_node_to_sim_receiver()

    print("\n" + "=" * 70)
    print("ALL NODE TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
