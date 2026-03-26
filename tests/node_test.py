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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Channels import Path
from Packet import RLNCPacket, RLNCType
from Node import Node
from Sender import SimSender


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

    for rlnc_type in [RLNCType.FEC, RLNCType.FB_FEC, RLNCType.CORRECTION]:
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
        assert sent.get_type() == RLNCType.CORRECTION, \
            f"Input {rlnc_type.name} should produce CORRECTION, got {sent.get_type().name}"
        assert set(sent.get_information_packets()) == {10, 20}

        # output channel
        out = get_output_channel_history(output_paths[0])
        assert len(out) == 1
        assert out[0].get_type() == RLNCType.CORRECTION, \
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
    inject_to_input(node, 3, make_rlnc([7, 8], global_path_id=4, rlnc_type=RLNCType.CORRECTION))

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
    expected_types = [RLNCType.NEW, RLNCType.NEW, RLNCType.CORRECTION, RLNCType.CORRECTION]
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
# N4 — 3 input epsilons, delivery ordering  (SimSender → NodeReceiver)
# ============================================================================

def test_N4_three_paths_delivery_ordering():
    """Wire a SimSender into the Node's receiver through 3 input paths with
    eps = [0, 0.3, 0.8].  After 200 steps the path-level receive counts
    must satisfy  path0 > path1 > path2."""
    print("\n=== Test N4: 3 paths eps=[0, 0.3, 0.8] — delivery ordering ===")

    PROP_DELAY = 2
    RTT = PROP_DELAY * 2
    EPSILONS = [0.0, 0.3, 0.8]
    NUM_STEPS = 200

    random.seed(42)

    input_paths = create_paths_with_epsilons(EPSILONS, PROP_DELAY, hop_index=0)
    output_paths = create_paths_with_epsilons([0.0] * 3, PROP_DELAY, hop_index=1)
    mock = MockNetwork(3)
    node = Node(
        hop_num=0,
        input_paths=input_paths,
        output_paths=output_paths,
        rtt=RTT,
        Network=mock,
    )

    sender = SimSender(
        num_of_packets_to_send=10_000,
        rtt=RTT,
        paths=input_paths,
        initial_epsilon=0.0,
        receiver=node.my_receiver,
        network=mock,
    )

    with redirect_stdout(open(os.devnull, "w")):
        for _ in range(NUM_STEPS):
            sender.run_step()

    counts = [
        len(node.my_receiver.get_receiver_path(i).get_received_channel_history())
        for i in range(3)
    ]

    print(f"  Received per path: {counts}")
    assert counts[0] > counts[1] > counts[2], \
        f"Expected path0 > path1 > path2, got {counts}"

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
# Main
# ============================================================================

def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING NODE TESTS")
    print("=" * 70)

    test_N1_single_path_new_packet_passes()
    test_N2_correction_types_pass_through()
    test_N3_four_paths_mixed_types()
    test_N4_three_paths_delivery_ordering()
    test_N5_output_info_matches_accumulated_buffer()
    test_N6_buffers_accumulate_across_steps()
    test_N7_natural_matching_sorts_by_r()
    test_N8_feedback_on_input_paths()

    print("\n" + "=" * 70)
    print("ALL NODE TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
