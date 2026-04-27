"""
Comprehensive tests for NodeReceiver class.

Test naming convention:
- NR1–NR5:   Correct saving of information packets to the appropriate buffer
- NR6–NR8:   Correct saving of the RLNC packet objects themselves
- NR9–NR11:  Correct mapping of global paths in curr_packet_type_in_glob_paths
- NR12–NR13: When no packet arrives, nothing is saved
- NR14:      SimSender → NodeReceiver integration smoke test (100 steps)
"""

import sys
import os
import random
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Channels import Path
from Packet import RLNCPacket, RLNCType, NodeRLNCType
from Receiver import NodeReceiver
from Sender import SimSender


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rlnc_packet(
    information_packets: list[int],
    global_path_id: int,
    rlnc_type: RLNCType = RLNCType.NEW,
    creation_time: int = 0,
) -> RLNCPacket:
    """Create an RLNCPacket that looks like it already arrived (prop_time=0)."""
    return RLNCPacket(
        global_path_id=global_path_id,
        type=rlnc_type,
        information_packets=information_packets,
        prop_time_left_in_channel=0,
        creation_time=creation_time,
    )


def create_paths(num_paths: int, prop_delay: int = 2, epsilon: float = 0.0) -> list[Path]:
    paths = []
    for i in range(num_paths):
        p = Path(
            propagation_delay=prop_delay,
            epsilon=epsilon,
            hop_index=0,
            path_index_in_hop=i,
        )
        p.set_global_path_index(i)
        paths.append(p)
    return paths


def create_node_receiver(
    num_paths: int = 2,
    prop_delay: int = 2,
    hop_num: int = 1,
) -> NodeReceiver:
    paths = create_paths(num_paths, prop_delay)
    rtt = prop_delay * 2
    return NodeReceiver(hop_num=hop_num, input_paths=paths, rtt=rtt)


def inject_to_arrived(receiver: NodeReceiver, path_index: int, packet: RLNCPacket):
    """Place a packet directly in the arrived buffer of the given path."""
    receiver.receiver_paths[path_index].forward_channel.arrived_packets.append(packet)


# ============================================================================
# 1. Correct saving of information packets
# ============================================================================

def test_NR1_new_packet_saves_info_to_new_buffer():
    """NEW packet's information packets go into new_information_packets_buffer."""
    print("\n=== Test NR1: NEW packet info -> new buffer ===")
    nr = create_node_receiver(num_paths=1)

    pkt = make_rlnc_packet([1, 2, 3], global_path_id=0, rlnc_type=RLNCType.NEW)
    inject_to_arrived(nr, 0, pkt)
    nr.run_step(time=1)

    assert nr.new_information_packets_buffer == {1, 2, 3}, \
        f"Expected {{1,2,3}}, got {nr.new_information_packets_buffer}"
    assert nr.correction_information_packets_buffer == set(), \
        f"Correction buffer should be empty, got {nr.correction_information_packets_buffer}"
    print("  PASSED")


def test_NR2_fec_packet_saves_info_to_correction_buffer():
    """FEC packet's information packets go into correction_information_packets_buffer."""
    print("\n=== Test NR2: FEC packet info -> correction buffer ===")
    nr = create_node_receiver(num_paths=1)

    pkt = make_rlnc_packet([4, 5], global_path_id=0, rlnc_type=RLNCType.FEC)
    inject_to_arrived(nr, 0, pkt)
    nr.run_step(time=1)

    assert nr.correction_information_packets_buffer == {4, 5}, \
        f"Expected {{4,5}}, got {nr.correction_information_packets_buffer}"
    assert nr.new_information_packets_buffer == set(), \
        f"New buffer should be empty, got {nr.new_information_packets_buffer}"
    print("  PASSED")


def test_NR3_fb_fec_and_correction_go_to_correction_buffer():
    """FB_FEC and CORRECTION packets both land in the correction buffer."""
    print("\n=== Test NR3: FB_FEC & CORRECTION -> correction buffer ===")
    nr = create_node_receiver(num_paths=2)

    fb_fec_pkt = make_rlnc_packet([10, 11], global_path_id=0, rlnc_type=RLNCType.FB_FEC)
    correction_pkt = make_rlnc_packet([12, 13], global_path_id=1, rlnc_type=NodeRLNCType.CORRECTION)

    inject_to_arrived(nr, 0, fb_fec_pkt)
    inject_to_arrived(nr, 1, correction_pkt)
    nr.run_step(time=1)

    assert nr.correction_information_packets_buffer == {10, 11, 12, 13}, \
        f"Expected {{10,11,12,13}}, got {nr.correction_information_packets_buffer}"
    assert nr.new_information_packets_buffer == set(), \
        f"New buffer should be empty, got {nr.new_information_packets_buffer}"
    print("  PASSED")


def test_NR4_multiple_new_packets_accumulate_as_set_union():
    """Multiple NEW packets across steps accumulate via set union."""
    print("\n=== Test NR4: Multiple NEW packets accumulate ===")
    nr = create_node_receiver(num_paths=1)

    pkt1 = make_rlnc_packet([1, 2], global_path_id=0, rlnc_type=RLNCType.NEW, creation_time=0)
    inject_to_arrived(nr, 0, pkt1)
    nr.run_step(time=1)

    pkt2 = make_rlnc_packet([2, 3, 4], global_path_id=0, rlnc_type=RLNCType.NEW, creation_time=1)
    inject_to_arrived(nr, 0, pkt2)
    nr.run_step(time=2)

    assert nr.new_information_packets_buffer == {1, 2, 3, 4}, \
        f"Expected {{1,2,3,4}}, got {nr.new_information_packets_buffer}"
    print("  PASSED")


def test_NR5_mixed_types_populate_correct_buffers():
    """NEW and correction packets across steps populate the right buffers independently."""
    print("\n=== Test NR5: Mixed types across steps ===")
    nr = create_node_receiver(num_paths=2)

    new_pkt = make_rlnc_packet([1, 2], global_path_id=0, rlnc_type=RLNCType.NEW, creation_time=0)
    inject_to_arrived(nr, 0, new_pkt)
    nr.run_step(time=1)

    fec_pkt = make_rlnc_packet([1, 2], global_path_id=1, rlnc_type=RLNCType.FEC, creation_time=1)
    inject_to_arrived(nr, 1, fec_pkt)
    nr.run_step(time=2)

    assert nr.new_information_packets_buffer == {1, 2}, \
        f"Expected new buffer {{1,2}}, got {nr.new_information_packets_buffer}"
    assert nr.correction_information_packets_buffer == {1, 2}, \
        f"Expected correction buffer {{1,2}}, got {nr.correction_information_packets_buffer}"
    print("  PASSED")


# ============================================================================
# 2. Correct saving of the packets themselves
# ============================================================================

def test_NR6_new_packet_stored_in_new_history():
    """NEW packet object is appended to new_rlnc_packets_history."""
    print("\n=== Test NR6: NEW packet -> new_rlnc_packets_history ===")
    nr = create_node_receiver(num_paths=1)

    pkt = make_rlnc_packet([1], global_path_id=0, rlnc_type=RLNCType.NEW)
    inject_to_arrived(nr, 0, pkt)
    nr.run_step(time=1)

    assert len(nr.new_rlnc_packets_history) == 1, \
        f"Expected 1 packet in new history, got {len(nr.new_rlnc_packets_history)}"
    assert nr.new_rlnc_packets_history[0] is pkt
    assert len(nr.correction_packets_history) == 0, \
        f"Correction history should be empty, got {len(nr.correction_packets_history)}"
    print("  PASSED")


def test_NR7_all_correction_types_stored_in_correction_history():
    """FEC, FB_FEC, and CORRECTION packets all go into correction_packets_history."""
    print("\n=== Test NR7: Correction packets -> correction_packets_history ===")
    nr = create_node_receiver(num_paths=3)

    fec = make_rlnc_packet([1], global_path_id=0, rlnc_type=RLNCType.FEC, creation_time=0)
    fb_fec = make_rlnc_packet([2], global_path_id=1, rlnc_type=RLNCType.FB_FEC, creation_time=0)
    correction = make_rlnc_packet([3], global_path_id=2, rlnc_type=NodeRLNCType.CORRECTION, creation_time=0)

    inject_to_arrived(nr, 0, fec)
    inject_to_arrived(nr, 1, fb_fec)
    inject_to_arrived(nr, 2, correction)
    nr.run_step(time=1)

    assert len(nr.correction_packets_history) == 3, \
        f"Expected 3, got {len(nr.correction_packets_history)}"
    assert nr.correction_packets_history[0] is fec
    assert nr.correction_packets_history[1] is fb_fec
    assert nr.correction_packets_history[2] is correction
    assert len(nr.new_rlnc_packets_history) == 0
    print("  PASSED")


def test_NR8_all_arrived_packets_in_general_receiver_history():
    """Every arrived packet (regardless of type) also appears in
    GeneralReceiver.received_rlnc_channel_history."""
    print("\n=== Test NR8: All packets in received_rlnc_channel_history ===")
    nr = create_node_receiver(num_paths=2)

    new_pkt = make_rlnc_packet([1], global_path_id=0, rlnc_type=RLNCType.NEW, creation_time=0)
    fec_pkt = make_rlnc_packet([2], global_path_id=1, rlnc_type=RLNCType.FEC, creation_time=0)

    inject_to_arrived(nr, 0, new_pkt)
    inject_to_arrived(nr, 1, fec_pkt)
    nr.run_step(time=1)

    history = nr.get_received_rlnc_channel_history()
    assert len(history) == 2, f"Expected 2 packets in history, got {len(history)}"
    assert new_pkt in history
    assert fec_pkt in history
    print("  PASSED")


# ============================================================================
# 3. Correct mapping of global paths in NodeReceiver
# ============================================================================

def test_NR9_global_path_mapping_single_new_packet():
    """Packet type is correctly mapped by its global_path_id."""
    print("\n=== Test NR9: Global path mapping – single path ===")
    nr = create_node_receiver(num_paths=1)

    pkt = make_rlnc_packet([1], global_path_id=0, rlnc_type=RLNCType.NEW)
    inject_to_arrived(nr, 0, pkt)
    nr.run_step(time=1)

    assert nr.curr_packet_type_in_glob_paths == {0: RLNCType.NEW}, \
        f"Expected {{0: NEW}}, got {nr.curr_packet_type_in_glob_paths}"
    print("  PASSED")


def test_NR10_global_path_mapping_multiple_types_on_multiple_paths():
    """Different packet types on different paths are all recorded correctly."""
    print("\n=== Test NR10: Global path mapping – multiple paths ===")
    nr = create_node_receiver(num_paths=3)

    new_pkt = make_rlnc_packet([1], global_path_id=0, rlnc_type=RLNCType.NEW, creation_time=0)
    fec_pkt = make_rlnc_packet([2], global_path_id=1, rlnc_type=RLNCType.FEC, creation_time=0)
    corr_pkt = make_rlnc_packet([3], global_path_id=2, rlnc_type=NodeRLNCType.CORRECTION, creation_time=0)

    inject_to_arrived(nr, 0, new_pkt)
    inject_to_arrived(nr, 1, fec_pkt)
    inject_to_arrived(nr, 2, corr_pkt)
    nr.run_step(time=1)

    expected = {0: RLNCType.NEW, 1: RLNCType.FEC, 2: NodeRLNCType.CORRECTION}
    assert nr.curr_packet_type_in_glob_paths == expected, \
        f"Expected {expected}, got {nr.curr_packet_type_in_glob_paths}"
    print("  PASSED")


def test_NR11_global_path_mapping_resets_each_step():
    """curr_packet_type_in_glob_paths is cleared at the start of every run_step."""
    print("\n=== Test NR11: Global path mapping resets each step ===")
    nr = create_node_receiver(num_paths=1)

    pkt = make_rlnc_packet([1], global_path_id=0, rlnc_type=RLNCType.NEW)
    inject_to_arrived(nr, 0, pkt)
    nr.run_step(time=1)
    assert len(nr.curr_packet_type_in_glob_paths) == 1, \
        "Mapping should have 1 entry after step with arrival"

    # Next step: no injection — mapping must be empty
    nr.run_step(time=2)
    assert nr.curr_packet_type_in_glob_paths == {}, \
        f"Expected empty mapping after step with no arrivals, got {nr.curr_packet_type_in_glob_paths}"
    print("  PASSED")


# ============================================================================
# 4. When packet does not arrive, no packet is saved
# ============================================================================

def test_NR12_no_arrivals_leaves_all_structures_empty():
    """Running several steps without injecting anything keeps every buffer and history empty."""
    print("\n=== Test NR12: No arrivals -> no state change ===")
    nr = create_node_receiver(num_paths=2)

    for t in range(1, 4):
        nr.run_step(time=t)

    assert nr.new_information_packets_buffer == set()
    assert nr.correction_information_packets_buffer == set()
    assert nr.new_rlnc_packets_history == []
    assert nr.correction_packets_history == []
    assert nr.get_received_rlnc_channel_history() == []
    assert nr.curr_packet_type_in_glob_paths == {}
    print("  PASSED")


def test_NR13_partial_arrivals_only_affected_paths_save():
    """With 3 paths, inject only to paths 0 and 2; path 1 must contribute nothing."""
    print("\n=== Test NR13: Partial arrivals – only receiving paths save ===")
    nr = create_node_receiver(num_paths=3)

    pkt0 = make_rlnc_packet([10, 11], global_path_id=0, rlnc_type=RLNCType.NEW, creation_time=0)
    pkt2 = make_rlnc_packet([20], global_path_id=2, rlnc_type=RLNCType.FEC, creation_time=0)

    inject_to_arrived(nr, 0, pkt0)
    # path 1: intentionally left empty
    inject_to_arrived(nr, 2, pkt2)
    nr.run_step(time=1)

    # Buffers
    assert nr.new_information_packets_buffer == {10, 11}, \
        f"Expected new buffer {{10,11}}, got {nr.new_information_packets_buffer}"
    assert nr.correction_information_packets_buffer == {20}, \
        f"Expected correction buffer {{20}}, got {nr.correction_information_packets_buffer}"

    # Packet histories
    assert len(nr.new_rlnc_packets_history) == 1
    assert nr.new_rlnc_packets_history[0] is pkt0
    assert len(nr.correction_packets_history) == 1
    assert nr.correction_packets_history[0] is pkt2

    # General history has both
    assert len(nr.get_received_rlnc_channel_history()) == 2

    # Global path mapping — path 1 absent
    assert nr.curr_packet_type_in_glob_paths == {0: RLNCType.NEW, 2: RLNCType.FEC}
    assert 1 not in nr.curr_packet_type_in_glob_paths
    print("  PASSED")


# ============================================================================
# 5. SimSender → NodeReceiver integration smoke test
# ============================================================================

def test_NR14_sim_sender_to_node_receiver_100_steps():
    """SimSender drives a NodeReceiver for 100 timesteps — nothing should crash.

    Setup mirrors MPNetwork: shared Path objects are wrapped by both
    SimSenderPath (inside SimSender) and ReceiverPath (inside NodeReceiver),
    so they share the same ForwardChannel / feedback Channel instances.
    """
    print("\n=== Test NR14: SimSender -> NodeReceiver 100 steps (smoke) ===")

    NUM_PATHS = 4
    PROP_DELAY = 2
    RTT = PROP_DELAY * 2
    EPSILONS = [0.1, 0.2, 0.1, 0.3]
    NUM_STEPS = 100

    random.seed(42)

    paths = [Path(PROP_DELAY, eps, 0, i) for i, eps in enumerate(EPSILONS)]
    for i, path in enumerate(paths):
        path.set_global_path_index(i)

    node_receiver = NodeReceiver(hop_num=1, input_paths=paths, rtt=RTT)
    sender = SimSender(
        num_of_packets_to_send=10_000,
        rtt=RTT,
        paths=paths,
        initial_epsilon=0.0,
        next_hop=node_receiver,
    )

    with redirect_stdout(open(os.devnull, "w")):
        for _ in range(NUM_STEPS):
            sender.run_step()

    # --- sanity checks: the simulation actually did something ---
    assert node_receiver.t > 0, "NodeReceiver time should have advanced"

    received = node_receiver.get_received_rlnc_channel_history()
    assert len(received) > 0, "NodeReceiver should have received at least some packets"

    new_count = len(node_receiver.new_rlnc_packets_history)
    corr_count = len(node_receiver.correction_packets_history)
    assert new_count + corr_count == len(received), \
        f"new({new_count}) + correction({corr_count}) != total received({len(received)})"

    assert len(node_receiver.new_information_packets_buffer) > 0, \
        "At least some NEW information packets should have been buffered"

    feedback_sent = node_receiver.get_sent_feedback_channel_history()
    assert len(feedback_sent) > 0, "NodeReceiver should have sent feedback"

    # Verify the sender actually received feedback through the feedback channels.
    # Feedback takes PROP_DELAY steps to travel from receiver to sender, so
    # the last PROP_DELAY steps' worth of feedback (NUM_PATHS per step) is
    # still in transit.
    sender_feedback_received = sum(
        len(p.all_feedback_history) for p in sender.paths
    )
    feedback_in_transit = NUM_PATHS * PROP_DELAY
    expected_sender_feedback = len(feedback_sent) - feedback_in_transit
    assert sender_feedback_received == expected_sender_feedback, \
        f"Sender should have received {expected_sender_feedback} feedbacks " \
        f"(sent={len(feedback_sent)} - in_transit={feedback_in_transit}), " \
        f"but got {sender_feedback_received}"
    assert sender_feedback_received > 0, "Sender must have received at least some feedback"

    print(f"  Completed {NUM_STEPS} steps without errors")
    print(f"  Receiver time:        {node_receiver.t}")
    print(f"  Packets received:     {len(received)}")
    print(f"  - NEW:                {new_count}")
    print(f"  - Correction:         {corr_count}")
    print(f"  New info buffer size: {len(node_receiver.new_information_packets_buffer)}")
    print(f"  Corr info buffer:     {len(node_receiver.correction_information_packets_buffer)}")
    print(f"  Feedback sent by receiver: {len(feedback_sent)}")
    print(f"  Feedback received by sender: {sender_feedback_received}")
    print(f"  Feedback in transit:  {feedback_in_transit}")
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING NODERECEIVER TESTS")
    print("=" * 70)

    # 1. Information-packet saving
    test_NR1_new_packet_saves_info_to_new_buffer()
    test_NR2_fec_packet_saves_info_to_correction_buffer()
    test_NR3_fb_fec_and_correction_go_to_correction_buffer()
    test_NR4_multiple_new_packets_accumulate_as_set_union()
    test_NR5_mixed_types_populate_correct_buffers()

    # 2. Packet-object saving
    test_NR6_new_packet_stored_in_new_history()
    test_NR7_all_correction_types_stored_in_correction_history()
    test_NR8_all_arrived_packets_in_general_receiver_history()

    # 3. Global path mapping
    test_NR9_global_path_mapping_single_new_packet()
    test_NR10_global_path_mapping_multiple_types_on_multiple_paths()
    test_NR11_global_path_mapping_resets_each_step()

    # 4. No arrivals
    test_NR12_no_arrivals_leaves_all_structures_empty()
    test_NR13_partial_arrivals_only_affected_paths_save()

    # 5. Integration smoke test
    test_NR14_sim_sender_to_node_receiver_100_steps()

    print("\n" + "=" * 70)
    print("ALL NODERECEIVER TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
