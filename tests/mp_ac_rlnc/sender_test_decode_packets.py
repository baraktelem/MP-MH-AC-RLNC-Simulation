"""
Tests for Sender.py - Testing sender's inference mechanism

This test file simulates receiver behavior to test how the sender
infers which packets the receiver has decoded based on ACK/NACK feedbacks.
"""
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Sender import SimSender
from Packet import FeedbackPacket, FeedbackType, RLNCType, PacketID
from Channels import Path
from CodedEquation import CodedEquation


def test_S1_infer_receiver_state_single_packet_ack():
    """
    Test S1: Single packet ACK decoding
    
    This test manually simulates receiver behavior by:
    - Injecting equations into sender.equations_waiting_feedback
    - Providing ACK feedbacks
    - Verifying sender correctly infers decoding
    
    Flow:
    1. Insert equation with [p1] to equations_waiting_feedback
    2. Give ACK for that equation  
    3. Verify p1 was decoded (in decoded_information_packets_history)
    """
    # Create a simple path for the sender
    PROP_DELAY = 5
    EPSILON = 0.0
    path = Path(propagation_delay=PROP_DELAY, epsilon=EPSILON, hop_index=0, path_index_in_hop=0)
    
    # Create sender
    NUM_PACKETS = 10
    RTT = 10
    # MAX_WINDOW = 5
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS,
        rtt=RTT,
        paths=[path]
    )
    
    # Step 1: Insert an equation to equations_waiting_feedback
    packet_id = 1
    related_rlnc_packet_id = PacketID(global_path_id=0, creation_time=0)
    information_packets = [packet_id]  # This equation contains only one information packet
    
    # Add to equations_waiting_feedback
    sender.equations_waiting_feedback[related_rlnc_packet_id] = CodedEquation(
        related_rlnc_packet_id=related_rlnc_packet_id,
        unknown_packets=information_packets
    )
    
    # Verify initial state
    assert related_rlnc_packet_id in sender.equations_waiting_feedback, \
        f"Equation {related_rlnc_packet_id} should be in equations_waiting_feedback"
    
    # Step 2: Give ACK on that packet
    ack_packet = FeedbackPacket(
        global_path_id=0,
        type=FeedbackType.ACK,
        related_packet_id=related_rlnc_packet_id,  # ACK is for the equation
        prop_time_left_in_channel=0,
        creation_time=10,
        related_information_packets=information_packets
    )
    
    # Add the ACK to sender's feedbacks
    sender.feedbacks = [ack_packet]
    
    # Call infer_receiver_state
    sender.infer_receiver_state()
    sender.t += 1  # Advance time
    
    # Step 3: Verify decoding happened
    assert related_rlnc_packet_id not in sender.equations_waiting_feedback, \
        f"Equation {related_rlnc_packet_id} should be removed from equations_waiting_feedback after decoding: {sender.equations_waiting_feedback}"

    assert related_rlnc_packet_id not in sender.acked_equations, \
        f"Equation {related_rlnc_packet_id} should be removed from acked_equations after decoding"

    assert packet_id in sender.decoded_information_packets_history, \
        f"Packet {packet_id} should be in decoded_information_packets_history after decoding"
    
    assert sender.oldest_information_packet_on_air == packet_id + 1, \
        f"oldest_information_packet_on_air should be {packet_id + 1}, got {sender.oldest_information_packet_on_air}"
    
    print("✓ Test S1 passed: Single packet was successfully decoded after ACK")


def test_S2_infer_receiver_state_multiple_packets():
    """
    Test S2: Multiple packet decoding (2 unknowns, 2 equations)
    
    This test verifies decoding triggers when sufficient equations accumulated.
    
    Flow:
    1. Inject equation 0 with [p1, p2]
    2. ACK equation 0 → moves to acked_equations (not enough to decode)
    3. Inject equation 1 with [p1, p2]
    4. ACK equation 1 → now 2 equations, 2 unknowns → DECODE!
    5. Verify p1 and p2 decoded, all equations cleared
    """
    print("\n=== Starting Test S2: Multiple Packet Decoding Test ===\n")
    # Create a simple path for the sender
    PROP_DELAY = 5
    EPSILON = 0.0
    path = Path(propagation_delay=PROP_DELAY, epsilon=EPSILON, hop_index=0, path_index_in_hop=0)
    
    # Create sender
    NUM_PACKETS = 10
    RTT = 10
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS,
        rtt=RTT,
        paths=[path]
    )
    
    # Step 1: Inject equation with packets [1, 2]
    related_rlnc_packet_id_0 = PacketID(global_path_id=0, creation_time=1)
    info_packets_0 = [1, 2]
    
    sender.equations_waiting_feedback[related_rlnc_packet_id_0] = CodedEquation(
        related_rlnc_packet_id=related_rlnc_packet_id_0,
        unknown_packets=info_packets_0.copy()
    )
    
    print(f"  ✓ Step 1 passed: Equation {related_rlnc_packet_id_0} injected with info_packets={info_packets_0}")
    
    # Step 2: ACK equation 0 and infer
    ack_0 = FeedbackPacket(
        global_path_id=0,
        type=FeedbackType.ACK,
        related_packet_id=related_rlnc_packet_id_0,
        prop_time_left_in_channel=0,
        creation_time=10,
        related_information_packets=info_packets_0.copy()
    )
    
    sender.feedbacks = [ack_0]
    sender.t += 1  # Advance time
    sender.infer_receiver_state()
    
    # Assert equation entered acked_equations (not enough equations to decode yet)
    assert related_rlnc_packet_id_0 in sender.acked_equations, \
        f"Equation {related_rlnc_packet_id_0} should be in acked_equations after ACK: {sender.acked_equations}"
    
    assert related_rlnc_packet_id_0 not in sender.equations_waiting_feedback, \
        f"Equation {related_rlnc_packet_id_0} should be removed from equations_waiting_feedback after ACK: {sender.equations_waiting_feedback}"
    
    # Not decoded yet (2 unknowns, 1 equation)
    assert len(sender.decoded_information_packets_history) == 0, \
        "Should not decode yet (2 unknowns, 1 equation)"
    
    print("  ✓ Step 2 passed: Equation 0 ACKed, moved to acked_equations, no decoding yet (2 unknowns, 1 equation)")
    
    # Step 3: Inject another equation with packets [1, 2]
    # This gives us 2 equations with 2 unknowns → should be decodable
    related_rlnc_packet_id_1 = PacketID(global_path_id=0, creation_time=2)
    info_packets_1 = [1, 2]
    
    sender.equations_waiting_feedback[related_rlnc_packet_id_1] = CodedEquation(
        related_rlnc_packet_id=related_rlnc_packet_id_1,
        unknown_packets=info_packets_1.copy()
    )
    
    print(f"  ✓ Step 3 passed: Equation {related_rlnc_packet_id_1} injected with info_packets={info_packets_1}")
    
    # Step 4: ACK equation 1 - now we have 2 equations for 2 unknowns → DECODABLE!
    ack_1 = FeedbackPacket(
        global_path_id=0,
        type=FeedbackType.ACK,
        related_packet_id=related_rlnc_packet_id_1,
        prop_time_left_in_channel=0,
        creation_time=11,
        related_information_packets=info_packets_1.copy()
    )
    
    sender.feedbacks = [ack_1]
    sender.t += 1  # Advance time
    sender.infer_receiver_state()
    
    print("  ✓ Step 4 passed: Equation 1 ACKed, should trigger decoding (2 unknowns, 2 equations)")
    
    # Step 5: Verify both packets were decoded
    assert 1 in sender.decoded_information_packets_history, \
        f"Packet 1 should be decoded, got {sender.decoded_information_packets_history}"
    assert 2 in sender.decoded_information_packets_history, \
        f"Packet 2 should be decoded, got {sender.decoded_information_packets_history}"
    
    # Both equations should be cleared after decoding
    assert related_rlnc_packet_id_0 not in sender.acked_equations, \
        "Equation 0 should be removed from acked_equations after decoding"
    assert related_rlnc_packet_id_1 not in sender.acked_equations, \
        "Equation 1 should be removed from acked_equations after decoding"
    
    assert related_rlnc_packet_id_0 not in sender.equations_waiting_feedback, \
        "Equation 0 should not be in equations_waiting_feedback"
    assert related_rlnc_packet_id_1 not in sender.equations_waiting_feedback, \
        "Equation 1 should be removed from equations_waiting_feedback"
    
    assert sender.oldest_information_packet_on_air == 3, \
        f"oldest_information_packet_on_air should be 3 (after decoding p1,p2), got {sender.oldest_information_packet_on_air}"

    print(f"  ✓ Test S2 passed: Both packets decoded!")
    print(f"     decoded_information_packets_history: {sender.decoded_information_packets_history}")

def test_S3_comprehensive_inference_with_acks_and_nacks():
    """
    Test S3: Comprehensive test with multiple ACKs, NACKs, and inference steps
    
    This test manually simulates receiver behavior to test complex scenarios:
    - Multiple equations (c1-c10) with overlapping information packets (p1-p6)
    - Mix of ACKs (innovative) and NACKs (non-innovative)
    - Sequential decoding: p1, then p2, then p3-p5 together
    - Equation cleanup after decoding
    
    Tests:
    - NACK removes equations from waiting
    - ACK moves equations to acked
    - Decoding triggers when unknowns <= equations
    - Decoded packets removed from remaining equations
    """
    # Setup
    PROP_DELAY = 5
    EPSILON = 0.0
    path = Path(propagation_delay=PROP_DELAY, epsilon=EPSILON, hop_index=0, path_index_in_hop=0)
    
    NUM_PACKETS = 20
    RTT = 10
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS,
        rtt=RTT,
        paths=[path]
    )
    
    print("\n=== Starting Test S3: Comprehensive Inference Test ===\n")
    
    # Step 1: Inject c1 with p1
    print("Step 1: Inject c1 with [p1]")
    c1_id = PacketID(global_path_id=0, creation_time=1)
    sender.equations_waiting_feedback[c1_id] = CodedEquation(related_rlnc_packet_id=c1_id, unknown_packets=[1])
    sender.t += 1
    sender.feedbacks = []
    sender.infer_receiver_state()
    assert c1_id in sender.equations_waiting_feedback
    print(f"  ✓ c1 in equations_waiting_feedback: {sender.equations_waiting_feedback}")
    
    # Step 2: Inject c2 with [p1,p2]
    print("\nStep 2: Inject c2 with [p1,p2]")
    c2_id = PacketID(global_path_id=0, creation_time=2)
    sender.equations_waiting_feedback[c2_id] = CodedEquation(related_rlnc_packet_id=c2_id, unknown_packets=[1, 2])
    sender.t += 1
    sender.feedbacks = []
    sender.infer_receiver_state()
    assert c1_id in sender.equations_waiting_feedback
    assert c2_id in sender.equations_waiting_feedback
    print(f"  ✓ c1,c2 in equations_waiting_feedback")
    
    # Step 3: Inject c3 with [p1,p2,p3]
    print("\nStep 3: Inject c3 with [p1,p2,p3]")
    c3_id = PacketID(global_path_id=0, creation_time=3)
    sender.equations_waiting_feedback[c3_id] = CodedEquation(related_rlnc_packet_id=c3_id, unknown_packets=[1, 2, 3])
    sender.t += 1
    sender.feedbacks = []
    sender.infer_receiver_state()
    assert c3_id in sender.equations_waiting_feedback
    print(f"  ✓ c1,c2,c3 in equations_waiting_feedback")
    
    # Step 4: Inject c4 with [p1,p2,p3]
    print("\nStep 4: Inject c4 with [p1,p2,p3]")
    c4_id = PacketID(global_path_id=0, creation_time=4)
    sender.equations_waiting_feedback[c4_id] = CodedEquation(related_rlnc_packet_id=c4_id, unknown_packets=[1, 2, 3])
    sender.t += 1
    sender.feedbacks = []
    sender.infer_receiver_state()
    assert c4_id in sender.equations_waiting_feedback
    print(f"  ✓ c1-c4 in equations_waiting_feedback")
    
    # Step 5: ACK c1 + Inject c5 (same time step)
    print("\nStep 5.1: ACK c1 - expect p1 to be decoded")
    sender.t += 1
    ack_c1 = FeedbackPacket(
        global_path_id=0,
        type=FeedbackType.ACK,
        related_packet_id=c1_id,
        prop_time_left_in_channel=0,
        creation_time=sender.t,
        related_information_packets=[1]
    )
    sender.feedbacks = [ack_c1]
    sender.infer_receiver_state()
    assert c1_id not in sender.equations_waiting_feedback, "c1 should be removed from equations_waiting_feedback"
    assert c1_id not in sender.acked_equations, "c1 should be decoded and removed from acked_equations"
    assert 1 in sender.decoded_information_packets_history, "p1 should be in decoded_information_packets_history"
    assert sender.oldest_information_packet_on_air == 2, f"oldest_information_packet_on_air should be 2, got {sender.oldest_information_packet_on_air}"
    print(f"  ✓ p1 decoded! decoded_information_packets_history: {sender.decoded_information_packets_history}")
    
    print("Step 5.2: Inject c5 with [p2,p3,p4] (same time step)")
    c5_id = PacketID(global_path_id=0, creation_time=5)
    sender.equations_waiting_feedback[c5_id] = CodedEquation(related_rlnc_packet_id=c5_id, unknown_packets=[2, 3, 4])
    assert c5_id in sender.equations_waiting_feedback
    print(f"  ✓ c5 added")
    
    # Step 6: ACK c2 + Inject c6 (same time step)
    print("\nStep 6.1: ACK c2 - expect p2 to be decoded")
    sender.t += 1
    ack_c2 = FeedbackPacket(
        global_path_id=0,
        type=FeedbackType.ACK,
        related_packet_id=c2_id,
        prop_time_left_in_channel=0,
        creation_time=sender.t,
        related_information_packets=[2]  # p1 already decoded, so trimmed to [p2]
    )
    sender.feedbacks = [ack_c2]
    sender.infer_receiver_state()
    assert c2_id not in sender.equations_waiting_feedback, "c2 should be removed from equations_waiting_feedback"
    assert c2_id not in sender.acked_equations, "c2 should be decoded and removed from acked_equations"
    assert 2 in sender.decoded_information_packets_history, "p2 should be in decoded_information_packets_history"
    assert sender.oldest_information_packet_on_air == 3, f"oldest_information_packet_on_air should be 3, got {sender.oldest_information_packet_on_air}"
    print(f"  ✓ p2 decoded! decoded_information_packets_history: {sender.decoded_information_packets_history}")
    
    print("Step 6.2: Inject c6 with [p3,p4,p5] (same time step)")
    c6_id = PacketID(global_path_id=0, creation_time=6)
    sender.equations_waiting_feedback[c6_id] = CodedEquation(related_rlnc_packet_id=c6_id, unknown_packets=[3, 4, 5])
    assert c6_id in sender.equations_waiting_feedback
    print(f"  ✓ c6 added")
    
    # Step 7: NACK c3 + Inject c7 (same time step)
    print("\nStep 7.1: NACK c3 - expect c3 removed from equations_waiting_feedback")
    sender.t += 1
    nack_c3 = FeedbackPacket(
        global_path_id=0,
        type=FeedbackType.NACK,
        related_packet_id=c3_id,
        prop_time_left_in_channel=0,
        creation_time=sender.t,
        related_information_packets=None
    )
    sender.feedbacks = [nack_c3]
    sender.infer_receiver_state()
    assert c3_id not in sender.equations_waiting_feedback, "c3 should be removed after NACK"
    assert c3_id not in sender.acked_equations, "c3 should not be in acked_equations"
    print(f"  ✓ c3 removed!")
    
    print("Step 7.2: Inject c7 with [p3,p4,p5] (same time step)")
    c7_id = PacketID(global_path_id=0, creation_time=7)
    sender.equations_waiting_feedback[c7_id] = CodedEquation(related_rlnc_packet_id=c7_id, unknown_packets=[3, 4, 5])
    assert c7_id in sender.equations_waiting_feedback
    print(f"  ✓ c7 added")
    
    # Step 8: NACK c4 + Inject c8 (same time step)
    print("\nStep 8.1: NACK c4 - expect c4 removed from equations_waiting_feedback")
    sender.t += 1
    nack_c4 = FeedbackPacket(
        global_path_id=0,
        type=FeedbackType.NACK,
        related_packet_id=c4_id,
        prop_time_left_in_channel=0,
        creation_time=sender.t,
        related_information_packets=None
    )
    sender.feedbacks = [nack_c4]
    sender.infer_receiver_state()
    assert c4_id not in sender.equations_waiting_feedback, "c4 should be removed after NACK"
    assert c4_id not in sender.acked_equations, "c4 should not be in acked_equations"
    print(f"  ✓ c4 removed!")
    
    print("Step 8.2: Inject c8 with [p3,p4,p5] (same time step)")
    c8_id = PacketID(global_path_id=0, creation_time=8)
    sender.equations_waiting_feedback[c8_id] = CodedEquation(related_rlnc_packet_id=c8_id, unknown_packets=[3, 4, 5])
    assert c8_id in sender.equations_waiting_feedback
    print(f"  ✓ c8 added")
    
    # Step 9: ACK c5 + Inject c9 (same time step)
    print("\nStep 9.1: ACK c5 - expect NO packets decoded (not enough equations)")
    sender.t += 1
    ack_c5 = FeedbackPacket(
        global_path_id=0,
        type=FeedbackType.ACK,
        related_packet_id=c5_id,
        prop_time_left_in_channel=0,
        creation_time=sender.t,
        related_information_packets=[3, 4]  # p1,p2 already decoded, so trimmed to [p3,p4]
    )
    sender.feedbacks = [ack_c5]
    prev_oldest = sender.oldest_information_packet_on_air
    sender.infer_receiver_state()
    assert c5_id not in sender.equations_waiting_feedback, "c5 should be moved to acked_equations"
    assert c5_id in sender.acked_equations, "c5 should be in acked_equations"
    assert sender.oldest_information_packet_on_air == prev_oldest, "No new packets should be decoded"
    print(f"  ✓ No decoding! acked_equations has {len(sender.acked_equations)} equations")
    
    print("Step 9.2: Inject c9 with [p3,p4,p5,p6] (same time step)")
    c9_id = PacketID(global_path_id=0, creation_time=9)
    sender.equations_waiting_feedback[c9_id] = CodedEquation(related_rlnc_packet_id=c9_id, unknown_packets=[3, 4, 5, 6])
    assert c9_id in sender.equations_waiting_feedback
    print(f"  ✓ c9 added with [p3,p4,p5,p6]")
    
    # Step 10: ACK c6 + Inject c10 (same time step)
    print("\nStep 10.1: ACK c6")
    sender.t += 1
    c6_id = PacketID(global_path_id=0, creation_time=6)
    ack_c6 = FeedbackPacket(
        global_path_id=0,
        type=FeedbackType.ACK,
        related_packet_id=c6_id,
        prop_time_left_in_channel=0,
        creation_time=sender.t,
        related_information_packets=[3, 4, 5]
    )
    sender.feedbacks = [ack_c6]
    sender.infer_receiver_state()
    assert c6_id not in sender.equations_waiting_feedback, "c6 should be moved to acked_equations"
    assert c6_id in sender.acked_equations, "c6 should be in acked_equations"
    print(f"  ✓ c6 ACKed! acked_equations has {len(sender.acked_equations)} equations")
    
    print("Step 10.2: Inject c10 with [p3,p4,p5,p6] (same time step)")
    c10_id = PacketID(global_path_id=0, creation_time=10)
    sender.equations_waiting_feedback[c10_id] = CodedEquation(related_rlnc_packet_id=c10_id, unknown_packets=[3, 4, 5, 6])
    assert c10_id in sender.equations_waiting_feedback
    print(f"  ✓ c10 added")
    
    print("Step 10.3: Inject c11 with [p6,p7] (same time step)")
    c11_id = PacketID(global_path_id=0, creation_time=11)
    sender.equations_waiting_feedback[c11_id] = CodedEquation(related_rlnc_packet_id=c11_id, unknown_packets=[6, 7])
    assert c11_id in sender.equations_waiting_feedback
    print(f"  ✓ c11 added")
    
    # Step 11: ACK c7 - check that p3,p4,p5 were decoded and acked_equations is empty
    print("\nStep 11: ACK c7 - expect p3,p4,p5 to be decoded (3 unknowns, 3 equations)")
    sender.t += 1
    ack_c7 = FeedbackPacket(
        global_path_id=0,
        type=FeedbackType.ACK,
        related_packet_id=c7_id,
        prop_time_left_in_channel=0,
        creation_time=sender.t,
        related_information_packets=[3, 4, 5]
    )
    sender.feedbacks = [ack_c7]
    sender.infer_receiver_state()
    
    # Assertions for Step 11
    assert c7_id not in sender.equations_waiting_feedback, "c7 should be removed from equations_waiting_feedback"
    assert len(sender.acked_equations) == 0, f"acked_equations should be empty after decoding, got {sender.acked_equations}"
    assert 3 in sender.decoded_information_packets_history, "p3 should be decoded"
    assert 4 in sender.decoded_information_packets_history, "p4 should be decoded"
    assert 5 in sender.decoded_information_packets_history, "p5 should be decoded"
    assert sender.oldest_information_packet_on_air == 6, f"oldest_information_packet_on_air should be 6 (after decoding p1-p5), got {sender.oldest_information_packet_on_air}"
    
    # Check that p3-p5 do not appear in any equation in equations_waiting_feedback
    for eq_id, equation in sender.equations_waiting_feedback.items():
        for pkt in [3, 4, 5]:
            assert pkt not in equation.get_unknown_packets(), \
                f"p{pkt} should not appear in equation {eq_id}, but found in {equation.get_unknown_packets()}"
    
    # Verify c9 was cleaned up: [3,4,5,6] → [6]
    assert c9_id in sender.equations_waiting_feedback, \
        f"c9 should still be in equations_waiting_feedback, got {sender.equations_waiting_feedback.keys()}"
    assert sender.equations_waiting_feedback[c9_id].get_unknown_packets() == [6], \
        f"c9 should contain [6] after cleanup (p3-p5 removed), got {sender.equations_waiting_feedback[c9_id].get_unknown_packets()}"
    
    # Verify c10 was cleaned up: [3,4,5,6] → [6]
    assert c10_id in sender.equations_waiting_feedback, \
        f"c10 should still be in equations_waiting_feedback, got {sender.equations_waiting_feedback.keys()}"
    assert sender.equations_waiting_feedback[c10_id].get_unknown_packets() == [6], \
        f"c10 should contain [6] after cleanup (p3-p5 removed), got {sender.equations_waiting_feedback[c10_id].get_unknown_packets()}"
    
    # Verify c11 still contains [6, 7] (no cleanup needed - didn't contain p3-p5)
    assert c11_id in sender.equations_waiting_feedback, \
        f"c11 should still be in equations_waiting_feedback, got {sender.equations_waiting_feedback.keys()}"
    assert sender.equations_waiting_feedback[c11_id].get_unknown_packets() == [6, 7], \
        f"c11 should contain [6, 7] (unchanged), got {sender.equations_waiting_feedback[c11_id].get_unknown_packets()}"
    
    print(f"  ✓ p3,p4,p5 decoded! decoded_information_packets_history: {sender.decoded_information_packets_history}")
    print(f"  ✓ acked_equations is empty: {sender.acked_equations}")
    print(f"  ✓ equations_waiting_feedback cleaned (p3-p5 removed from remaining equations)")
    print(f"  ✓ c9 now contains [6] after cleanup (was [3,4,5,6])")
    print(f"  ✓ c10 now contains [6] after cleanup (was [3,4,5,6])")
    print(f"  ✓ c11 still contains [6, 7] (unchanged)")
    
    print("\n=== Test S3 Summary ===")
    print(f"Final decoded_information_packets_history: {sender.decoded_information_packets_history}")
    print(f"Final oldest_information_packet_on_air: {sender.oldest_information_packet_on_air}")
    print(f"Final equations_waiting_feedback: {len(sender.equations_waiting_feedback)} equations")
    print(f"  - c9: {sender.equations_waiting_feedback[c9_id].get_unknown_packets()}")
    print(f"  - c10: {sender.equations_waiting_feedback[c10_id].get_unknown_packets()}")
    print(f"  - c11: {sender.equations_waiting_feedback[c11_id].get_unknown_packets()}")
    print(f"Final acked_equations: {sender.acked_equations}")
    print("\n✓ Test S3 passed: Comprehensive inference test completed successfully!")


if __name__ == "__main__":
    test_S1_infer_receiver_state_single_packet_ack()
    test_S2_infer_receiver_state_multiple_packets()
    test_S3_comprehensive_inference_with_acks_and_nacks()
    print("\n✓ All sender tests passed!")

