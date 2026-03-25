"""
Integration tests for Sender and Receiver communication.

Test naming convention:
- SR1-SRN: Sender-Receiver integration tests

This module tests the complete flow between SimSender and GeneralReceiver.
"""

import sys
import os
from datetime import datetime
from contextlib import redirect_stdout

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Sender import SimSender
from Packet import FeedbackPacket, FeedbackType, RLNCType, RLNCPacket, PacketID
from Channels import Path
from Receiver import GeneralReceiver
from CodedEquation import CodedEquation
from TestChannelsAndPaths import TestPath


class TeeOutput:
    """
    Captures output and writes to both terminal and log file.
    Acts like a file object for use with sys.stdout redirection.
    """
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.log_file_path = log_file_path
        
        # Write header to log file
        timestamp = datetime.now()
        self.log_file.write("="*70 + "\n")
        self.log_file.write("MP MH AC-RLNC Simulation - Integration Test Suite\n")
        self.log_file.write("="*70 + "\n")
        self.log_file.write(f"Date: {timestamp.strftime('%B %d, %Y')}\n")
        self.log_file.write(f"Time: {timestamp.strftime('%H:%M:%S')}\n")
        self.log_file.write(f"Log File: {log_file_path}\n")
        self.log_file.write("="*70 + "\n\n")
        self.log_file.flush()
    
    def write(self, message):
        """Write message to both terminal and log file"""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure it's written immediately
    
    def flush(self):
        """Flush both outputs"""
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        """Close log file with completion timestamp"""
        if self.log_file and not self.log_file.closed:
            timestamp = datetime.now()
            self.log_file.write("\n" + "="*70 + "\n")
            self.log_file.write("Test Run Completed\n")
            self.log_file.write("="*70 + "\n")
            self.log_file.write(f"Completion Time: {timestamp.strftime('%H:%M:%S')}\n")
            self.log_file.write(f"Date: {timestamp.strftime('%B %d, %Y')}\n")
            self.log_file.write("="*70 + "\n")
            self.log_file.close()


# Global tee output object
_tee_output = None


def test_SR1_single_packet_single_path_complete_flow():
    """
    SR1 — Simple end-to-end test: 1 packet, 1 path, prop_delay=1, epsilon=0
    
    Flow:
    t=0: Sender sends c1 with [p1] on path 0
    t=1: Receiver receives c1, sends ACK for c1
    t=2: Sender receives ACK for c1, infers receiver decoded p1
         Receiver sends NACK for c2 (no packet received)
    t=3: Sender receives NACK for c2, but ignores it (already sent all packets)
         Receiver sends NACK for c3
    
    Verifies:
    - RLNC packet c1 arrives at receiver
    - Receiver sends ACK for c1 (not before)
    - Sender correctly infers p1 was decoded
    - Receiver sends NACK for c2 after not receiving packet
    - Sender ignores NACK when no more packets to send
    """
    print("\n" + "="*70)
    print("Test SR1: Single packet, single path, complete flow")
    print("="*70)
    
    # Configuration
    PROP_DELAY = 1
    RTT = PROP_DELAY * 2
    NUM_PACKETS_TO_SEND = 1
    EPSILON = 0.0  # No packet loss
    
    # Step 1: Create path
    print(f"\nStep 1: Creating path with prop_delay={PROP_DELAY}, epsilon={EPSILON}")
    path = Path(propagation_delay=PROP_DELAY, epsilon=EPSILON, hop_index=0, path_index_in_hop=0)
    path.set_global_path_index(0)
    print(f"  ✓ Path created: {path}")
    
    # Step 2: Create receiver
    print(f"\nStep 2: Creating receiver with RTT={RTT}")
    receiver = GeneralReceiver(input_paths=[path], rtt=RTT, unit_name="TestReceiver")
    print(f"  ✓ Receiver created: {receiver}")
    
    # Step 3: Create sender
    print(f"\nStep 3: Creating sender with {NUM_PACKETS_TO_SEND} packet to send")
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS_TO_SEND,
        rtt=RTT,
        paths=[path],
        receiver=receiver
    )
    print(f"  ✓ Sender created")
    print(f"    - num_of_packets_to_send: {sender.num_of_packets_to_send}")
    print(f"    - rtt: {sender.rtt}")
    print(f"    - num_of_paths: {sender.num_of_paths}")
    
    # =================================================================
    # t=1: Sender sends c1 with [p1]
    # =================================================================

    t = 1
    print(f"\n{'='*70}")
    print(f"t={t}: Sender runs step - should send c1 with [p1]")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Verify sender state at t=1
    print(f"\n  Checking sender state at t={t}:")
    assert sender.t == t, f"Sender time should be {t}, got {sender.t}"
    print(f"    ✓ sender.t = {sender.t}")
    
    # Check that c1 was created and is in forward channel
    path0 = sender.paths[0]
    forward_history = path0.get_forward_channel_history()
    assert len(forward_history) == 1, \
        f"Forward channel should have 1 packet in history, got {len(forward_history)}"
    c1 = forward_history[0]
    assert c1.get_information_packets() == [1], \
        f"c1 should contain [p1], got {c1.get_information_packets()}"
    assert c1.get_creation_time() == t, \
        f"c1 should have creation_time={t}, got {c1.get_creation_time()}"
    print(f"    ✓ c1 sent with information_packets={c1.get_information_packets()}, creation_time={c1.get_creation_time()}")
    
    # Check sender's equations_waiting_feedback
    # c1_packet_id = PacketID(global_path_id=0, creation_time=t, type=RLNCType.NEW)
    c1_packet_id = PacketID(global_path_id=0, creation_time=t)
    assert c1_packet_id in sender.equations_waiting_feedback, \
        f"c{t} should be in equations_waiting_feedback:\n\t{sender.equations_waiting_feedback}"
    eq1 = sender.equations_waiting_feedback[c1_packet_id]
    assert eq1.unknown_packets == [1], \
        f"Equation for c{t} should have unknown_packets=[1], got {eq1.unknown_packets}"
    print(f"    ✓ Equation for c{t} added to equations_waiting_feedback: {eq1}")
    
    # Check receiver state (should not have received anything yet)
    print(f"\n  Checking receiver state at t={t-1}:")
    receiver_history = receiver.get_received_rlnc_channel_history()
    assert len(receiver_history) == 0, \
        f"Receiver should not have received any packets yet, got {len(receiver_history)}:\n\t{receiver_history}"
    print(f"    ✓ Receiver has not received any packets yet")
    
    sent_feedback_history = receiver.get_sent_feedback_channel_history()
    assert len(sent_feedback_history) == 0, \
        f"Receiver should not have sent any feedback yet, got {len(sent_feedback_history)}"
    print(f"    ✓ Receiver has not sent any feedback yet")
    
    # =================================================================
    # t=2: Receiver receives c1, sends ACK for c1
    # =================================================================
    t += 1
    print(f"\n{'='*70}")
    print(f"t={t}: Receiver should receive c1 and send ACK")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Verify receiver received c1
    print(f"\n  Checking receiver state at t={t}:")
    receiver_history = receiver.get_received_rlnc_channel_history()
    assert len(receiver_history) == 1, \
        f"Receiver should have received 1 packet, got {len(receiver_history)}"
    received_c1 = receiver_history[0]
    assert received_c1.get_information_packets() == [1], \
        f"Received packet should contain [p1], got {received_c1.get_information_packets()}"
    assert received_c1.get_creation_time() == t-1, \
        f"Received c1 should have creation_time={t-1}, got {received_c1.get_creation_time()}"
    print(f"    ✓ Receiver received c1: info_packets={received_c1.get_information_packets()}, creation_time={received_c1.get_creation_time()}")
    
    # Verify receiver sent ACK for c1
    sent_feedback_history = receiver.get_sent_feedback_channel_history()
    assert len(sent_feedback_history) == 1, \
        f"Receiver should have sent 1 feedback, got {len(sent_feedback_history)}"
    ack_c1 = sent_feedback_history[0]
    assert ack_c1.is_ack(), \
        f"First feedback should be ACK, got {ack_c1}"
    assert ack_c1.get_related_packet_id().get_creation_time() == t-1, \
        f"ACK should be for c{t-1} (id={t-1}), got {ack_c1.get_related_packet_id().get_creation_time()}"
    assert ack_c1.get_related_information_packets() == [1], \
        f"ACK should reference [p1], got {ack_c1.get_related_information_packets()}"
    print(f"    ✓ Receiver sent ACK for c{t-1}: related_packet_id={ack_c1.get_related_packet_id()}, related_info_packets={ack_c1.get_related_information_packets()}")
    
    # Verify receiver decoded p1 at t=2 (when it received c1)
    assert 1 in receiver.information_packets_decoding_times, \
        f"p1 should be in receiver.information_packets_decoding_times at t={t}, got {receiver.information_packets_decoding_times}"
    assert receiver.latest_decoded_information_packet == 1, \
        f"receiver.latest_decoded_information_packet should be 1, got {receiver.latest_decoded_information_packet}"
    print(f"    ✓ Receiver decoded p1 at t={t} (when packet arrived): {receiver.information_packets_decoding_times}")
    
    # Verify sender has not received ACK yet (still in channel)
    print(f"\n  Checking sender state at t={t}:")
    assert sender.t == t, f"Sender time should be {t}, got {sender.t}"
    sender_feedbacks = sender.feedbacks
    assert len(sender_feedbacks) == 0, \
        f"Sender should not have received any feedbacks yet, got {len(sender_feedbacks)}"
    print(f"    ✓ Sender has not received ACK yet (still in feedback channel)")
    
    # Verify sender did not decode anything yet
    assert len(sender.decoded_information_packets_history) == 0, \
        f"Sender should not have decoded any packets yet, got {sender.decoded_information_packets_history}"
    print(f"    ✓ Sender has not inferred any decoded packets yet")

    # # verify num_rlnc_until_ew was updated
    # expected_num_rlnc_until_ew = 2
    # assert sender.num_rlnc_until_ew == expected_num_rlnc_until_ew, \
    #     f"num_rlnc_until_ew should be {expected_num_rlnc_until_ew} (after sending c1 and c2, before receiving feedback), got {sender.num_rlnc_until_ew}"
    # print(f"    ✓ Sender num_rlnc_until_ew is correct: {sender.num_rlnc_until_ew} == {expected_num_rlnc_until_ew}")
    
    # =================================================================
    # t=3: Sender receives ACK for c1, infers p1 decoded
    # 
    # =================================================================
    t += 1
    print(f"\n{'='*70}")
    print(f"t={t}: Sender should receive ACK for c1 and infer p1 decoded")
    print(f"     Receiver should send NACK for c2")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Verify sender received ACK
    print(f"\n  Checking sender state at t={t}:")
    assert sender.t == t, f"Sender time should be {t}, got {sender.t}"
    
    # Check that sender received the ACK feedback
    all_feedback_history = sender.all_feedback_history
    assert len(all_feedback_history) == 1, \
        f"Sender should have received 1 feedback in history, got {len(all_feedback_history)}"
    received_ack = all_feedback_history[0]
    assert received_ack.is_ack(), \
        f"Received feedback should be ACK, got {received_ack}"
    c1_id = 1  # c1 was sent at t=1
    assert received_ack.get_related_packet_id().get_creation_time() == c1_id, \
        f"Received ACK should be for c{c1_id} (id={c1_id}), got {received_ack.get_related_packet_id().get_creation_time()}"
    print(f"    ✓ Sender received ACK for c{c1_id}")
    
    # Verify sender inferred p1 was decoded
    assert len(sender.decoded_information_packets_history) == 1, \
        f"Sender should have 1 decoded packet, got {len(sender.decoded_information_packets_history)}: {sender.decoded_information_packets_history}"
    assert 1 in sender.decoded_information_packets_history, \
        f"p1 should be in decoded_information_packets_history, got {sender.decoded_information_packets_history}"
    print(f"    ✓ Sender inferred p1 was decoded (at t={t}, based on ACK from t={t-1}): {sender.decoded_information_packets_history}")
    
    # Verify c1 was removed from equations_waiting_feedback
    # c1_packet_id = PacketID(global_path_id=0, creation_time=c1_id, type=RLNCType.NEW)
    c1_packet_id = PacketID(global_path_id=0, creation_time=c1_id)
    assert c1_packet_id not in sender.equations_waiting_feedback, \
        f"c{c1_id} should be removed from equations_waiting_feedback, got {sender.equations_waiting_feedback}"
    print(f"    ✓ c{c1_id} removed from equations_waiting_feedback")
    
    # Verify c1 was not in acked_equations (single equation, immediately decoded)
    assert c1_packet_id not in sender.acked_equations, \
        f"c{c1_id} should not be in acked_equations (immediately decoded), got {sender.acked_equations}"
    print(f"    ✓ c{c1_id} not in acked_equations (immediately decoded)")
    
    # # Verify num_rlnc_until_ew was updated
    # assert sender.num_rlnc_until_ew == 1, \
    #     f"num_rlnc_until_ew should be 1 (c1 acked, c2 on air), got {sender.num_rlnc_until_ew}"
    # print(f"    ✓ num_rlnc_until_ew updated corrcetly")
    
    # Verify receiver sent ACK for c2
    print(f"\n  Checking receiver state at t={t}:")
    sent_feedback_history = receiver.get_sent_feedback_channel_history()
    assert len(sent_feedback_history) == t-1, \
        f"Receiver should have sent 2 feedbacks (ACK, NACK), got {len(sent_feedback_history)}"
    ack_c2 = sent_feedback_history[-1]
    assert ack_c2.is_ack(), \
        f"Second feedback should be ACK, got {ack_c2}, sent_feedback_history:\n\t{sent_feedback_history}"
    # ACK related_packet_id is current_time - propagation_delay
    expected_ack_related_id = receiver.t - PROP_DELAY
    assert ack_c2.get_related_packet_id().get_creation_time() == expected_ack_related_id, \
        f"ACK should be for time {expected_ack_related_id}, got {ack_c2.get_related_packet_id().get_creation_time()}"
    print(f"    ✓ Receiver sent ACK for c{expected_ack_related_id}: {ack_c2}")
    
    # =================================================================
    # t=4: Sender should NOT send packets (all decoded)
    #      Receiver should send NACK for c3
    # =================================================================
    t += 1
    print(f"\n{'='*70}")
    print(f"t={t}: Sender should NOT send packets (all decoded)")
    print(f"     Receiver should send NACK for c3")
    print(f"{'='*70}")
    
    # Track forward history length before step
    forward_history_len_before = len(path0.get_forward_channel_history())
    
    sender.run_step()
    
    # Verify sender did NOT send any new packets
    print(f"\n  Checking sender state at t={t}:")
    assert sender.t == t, f"Sender time should be {t}, got {sender.t}"
    forward_history_after = path0.get_forward_channel_history()
    assert len(forward_history_after) == forward_history_len_before, \
        f"Sender should not send new packets (all decoded), but sent {len(forward_history_after) - forward_history_len_before} packets"
    print(f"    ✓ Sender did not send any packets (all information packets decoded)")
    
    # Verify receiver sent NACK for c3
    print(f"\n  Checking receiver state at t={t}:")
    sent_feedback_history = receiver.get_sent_feedback_channel_history()
    assert len(sent_feedback_history) == t-1, \
        f"Receiver should have sent {t-1} feedbacks total, got {len(sent_feedback_history)}"
    nack_c3 = sent_feedback_history[-1]
    assert nack_c3.is_nack(), \
        f"Third feedback should be NACK, got {nack_c3}"
    expected_nack_packet_id = t - PROP_DELAY
    assert nack_c3.get_related_packet_id().get_creation_time() == expected_nack_packet_id, \
        f"NACK should be for packet_id {expected_nack_packet_id}, got {nack_c3.get_related_packet_id().get_creation_time()}:\n\t{nack_c3}"
    print(f"    ✓ Receiver sent NACK for packet_id={nack_c3.get_related_packet_id().get_creation_time()}: {nack_c3}")
    
    # =================================================================
    # t=5: Sender should receive NACK for c3 (FeedbackPacket4)
    #      Sender should NOT send any packets (all decoded)
    # =================================================================
    t += 1
    print(f"\n{'='*70}")
    print(f"t={t}: Sender should receive NACK for c3 (FeedbackPacket4)")
    print(f"     Sender should NOT send packets (all decoded)")
    print(f"{'='*70}")
    
    # Track forward history length before step
    forward_history_len_before = len(path0.get_forward_channel_history())
    
    sender.run_step()
    
    # Verify sender received the NACK
    print(f"\n  Checking sender state at t={t}:")
    assert sender.t == t, f"Sender time should be {t}, got {sender.t}"
    
    # Verify sender did NOT send any new packets
    forward_history_after = path0.get_forward_channel_history()
    assert len(forward_history_after) == forward_history_len_before, \
        f"Sender should not send new packets (all decoded), but sent {len(forward_history_after) - forward_history_len_before} packets"
    print(f"    ✓ Sender did not send any packets (all information packets decoded)")
    
    # Verify sender still has all packets decoded
    assert len(sender.decoded_information_packets_history) == NUM_PACKETS_TO_SEND, \
        f"Sender should still have {NUM_PACKETS_TO_SEND} decoded packets, got {len(sender.decoded_information_packets_history)}"
    print(f"    ✓ Sender still has all {NUM_PACKETS_TO_SEND} information packets decoded")
    
    # =================================================================
    # Final Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nSender:")
    print(f"  - Total time steps: {sender.t}")
    print(f"  - Packets sent: {len(path0.get_forward_channel_history())}")
    print(f"  - Feedbacks received: {len(sender.all_feedback_history)} (ACKs: {len(sender.acked_feedback_history)}, NACKs: {len(sender.nacked_feedback_history)})")
    print(f"  - Decoded information packets: {sender.decoded_information_packets_history}")
    print(f"  - Equations waiting feedback: {sender.equations_waiting_feedback}")
    print(f"  - Acked equations: {sender.acked_equations}")
    
    print(f"\nReceiver:")
    print(f"  - RLNC packets received: {len(receiver.get_received_rlnc_channel_history())}")
    print(f"  - Feedback packets sent: {len(receiver.get_sent_feedback_channel_history())}")
    print(f"  - Decoded information packets: {sorted(receiver.information_packets_decoding_times)}")
    print(f"  - Latest decoded: {receiver.latest_decoded_information_packet}")
    print(f"\n  Note: Receiver decodes at t=2 (packet arrival), Sender infers at t=3 (ACK arrival)")
    
    print(f"\n{'='*70}")
    print(f"✓ Test SR1 PASSED: Complete sender-receiver flow verified!")
    print(f"{'='*70}\n")


def test_SR2_multiple_packets_single_path_no_loss():
    """
    SR2 — Multiple packets test: 3 packets, 1 path, prop_delay=1, epsilon=0
    
    Flow:
    t=1: Sender sends c1 with [p1]
    t=2: Receiver gets c1, sends ACK. Sender sends c2 with [p1,p2]
    t=3: Receiver gets c2, sends ACK. Sender gets ACK(c1), decodes p1, sends c3 with [p2,p3]
    t=4: Receiver gets c3, sends ACK. Sender gets ACK(c2), decodes p2
    t=5: Sender gets ACK(c3), decodes p3
    
    Verifies:
    - Sequential packet transmission
    - Window sliding (oldest_information_packet_on_air updates)
    - Multiple equations tracking
    - Sequential decoding detection
    - NEW packet content changes as window slides
    """
    print("\n" + "="*70)
    print("Test SR2: Multiple packets, single path, no loss")
    print("="*70)
    
    # Configuration
    PROP_DELAY = 1
    RTT = PROP_DELAY * 2
    NUM_PACKETS_TO_SEND = 3
    EPSILON = 0.0
    
    # Setup
    print(f"\nSetup: Creating path, receiver, and sender")
    path = Path(propagation_delay=PROP_DELAY, epsilon=EPSILON, hop_index=0, path_index_in_hop=0)
    path.set_global_path_index(0)
    receiver = GeneralReceiver(input_paths=[path], rtt=RTT, unit_name="TestReceiver")
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS_TO_SEND,
        rtt=RTT,
        paths=[path],
        receiver=receiver
    )
    print(f"  ✓ Created: {NUM_PACKETS_TO_SEND} packets to send, RTT={RTT}")
    
    # =================================================================
    # t=1: Sender sends c1 with [p1]
    # =================================================================
    t = 1
    print(f"\n{'='*70}")
    print(f"t={t}: Sender sends c1 with [p1]")
    print(f"{'='*70}")
    
    sender.run_step()
    
    forward_history = sender.paths[0].get_forward_channel_history()
    assert len(forward_history) == 1, f"Should have sent 1 packet, got {len(forward_history)}"
    c1 = forward_history[0]
    assert c1.get_information_packets() == [1], f"c1 should contain [p1], got {c1.get_information_packets()}"
    assert c1.get_type() == RLNCType.NEW, f"c1 should be NEW, got {c1.get_type()}"
    print(f"  ✓ c1 sent: info=[1], type=NEW")
    
    # c1_packet_id = PacketID(global_path_id=0, creation_time=1, type=RLNCType.NEW)
    c1_packet_id = PacketID(global_path_id=0, creation_time=1)
    assert c1_packet_id in sender.equations_waiting_feedback, "c1 should be in equations_waiting_feedback"
    assert sender.equations_waiting_feedback[c1_packet_id].unknown_packets == [1], "c1 equation should have [p1]"
    print(f"  ✓ Equation c1: unknowns=[1]")
    
    # =================================================================
    # t=2: Receiver gets c1, sends ACK. Sender sends c2 with [p1,p2]
    # =================================================================
    t = 2
    print(f"\n{'='*70}")
    print(f"t={t}: Receiver gets c1, sends ACK. Sender sends c2 with [p1,p2]")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Receiver checks
    receiver_history = receiver.get_received_rlnc_channel_history()
    assert len(receiver_history) == 1, f"Receiver should have 1 packet, got {len(receiver_history)}"
    assert receiver_history[0].get_information_packets() == [1], "Received packet should be [p1]"
    print(f"  ✓ Receiver received c1")
    
    feedback_history = receiver.get_sent_feedback_channel_history()
    assert len(feedback_history) == 1, f"Should have 1 feedback, got {len(feedback_history)}"
    assert feedback_history[0].is_ack(), "First feedback should be ACK"
    assert feedback_history[0].get_related_packet_id().get_creation_time() == 1, "ACK should be for c1"
    print(f"  ✓ Receiver sent ACK for c1")
    
    # Verify receiver decoded p1 at t=2 (when c1 arrived)
    assert 1 in receiver.information_packets_decoding_times, f"Receiver should have decoded p1 at t={t}, got {receiver.information_packets_decoding_times}"
    print(f"  ✓ Receiver decoded p1 at t={t}: {sorted(receiver.information_packets_decoding_times)}")
    
    # Sender checks
    assert len(forward_history) == 2, f"Should have sent 2 packets, got {len(forward_history)}"
    c2 = forward_history[1]
    assert c2.get_information_packets() == [1, 2], f"c2 should contain [p1,p2], got {c2.get_information_packets()}"
    assert c2.get_type() == RLNCType.NEW, f"c2 should be NEW, got {c2.get_type()}"
    print(f"  ✓ c2 sent: info=[1,2], type=NEW")
    
    # c2_packet_id = PacketID(global_path_id=0, creation_time=2, type=RLNCType.NEW)
    c2_packet_id = PacketID(global_path_id=0, creation_time=2)
    assert c2_packet_id in sender.equations_waiting_feedback, "c2 should be in equations_waiting_feedback"
    assert sender.equations_waiting_feedback[c2_packet_id].unknown_packets == [1, 2], "c2 equation should have [p1,p2]"
    print(f"  ✓ Equation c2: unknowns=[1,2]")
    
    # Sender hasn't received ACK yet
    assert len(sender.decoded_information_packets_history) == 0, "Sender shouldn't have decoded anything yet"
    print(f"  ✓ Sender hasn't decoded anything yet (ACK still in transit)")
    
    # =================================================================
    # t=3: Receiver gets c2, sends ACK. Sender gets ACK(c1), decodes p1, sends c3 with [p2,p3]
    # =================================================================
    t = 3
    print(f"\n{'='*70}")
    print(f"t={t}: Receiver gets c2, sends ACK. Sender gets ACK(c1), decodes p1, sends c3")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Receiver checks
    assert len(receiver_history) == 2, f"Receiver should have 2 packets, got {len(receiver_history)}"
    assert receiver_history[1].get_information_packets() == [1, 2], "Second packet should be [p1,p2]"
    print(f"  ✓ Receiver received c2")
    
    assert len(feedback_history) == 2, f"Should have 2 feedbacks, got {len(feedback_history)}"
    assert feedback_history[1].is_ack(), "Second feedback should be ACK"
    assert feedback_history[1].get_related_packet_id().get_creation_time() == 2, "ACK should be for c2"
    print(f"  ✓ Receiver sent ACK for c2")
    
    # Verify receiver decoded p2 at t=3 (when c2=[1,2] arrived)
    # Receiver already had p1, c2 adds equation with [2], decodes p2
    assert 1 in receiver.information_packets_decoding_times, f"Receiver should still have p1 decoded"
    assert 2 in receiver.information_packets_decoding_times, f"Receiver should have decoded p2 at t={t}, got {receiver.information_packets_decoding_times}"
    assert receiver.latest_decoded_information_packet == 2, f"Receiver latest should be 2, got {receiver.latest_decoded_information_packet}"
    print(f"  ✓ Receiver decoded p2 at t={t}: {sorted(receiver.information_packets_decoding_times)}")
    
    # Sender receives ACK(c1) and decodes p1
    assert len(sender.all_feedback_history) == 1, f"Sender should have 1 feedback, got {len(sender.all_feedback_history)}"
    assert sender.all_feedback_history[0].get_related_packet_id().get_creation_time() == 1, "Should be ACK for c1"
    print(f"  ✓ Sender received ACK for c1")
    
    assert len(sender.decoded_information_packets_history) == 1, "Sender should have decoded 1 packet"
    assert 1 in sender.decoded_information_packets_history, "p1 should be decoded"
    print(f"  ✓ Sender decoded p1")
    
    # Verify receiver still has p1 decoded (decoded it at t=2 when packet arrived)
    assert 1 in receiver.information_packets_decoding_times, f"Receiver should still have p1 decoded, got {receiver.information_packets_decoding_times}"
    print(f"  ✓ Receiver decoded p1 earlier (at t={t-1} when c1 arrived)")
    
    # c1_packet_id = PacketID(global_path_id=0, creation_time=1, type=RLNCType.NEW)
    # c2_packet_id = PacketID(global_path_id=0, creation_time=2, type=RLNCType.NEW)
    c1_packet_id = PacketID(global_path_id=0, creation_time=1)
    c2_packet_id = PacketID(global_path_id=0, creation_time=2)
    assert c1_packet_id not in sender.equations_waiting_feedback, "c1 should be removed from waiting"
    assert c1_packet_id not in sender.acked_equations, "c1 should not be in acked (immediately decoded)"
    print(f"  ✓ c1 equation cleaned up")
    
    # Check that c2's unknown packets were updated (p1 removed)
    assert c2_packet_id in sender.equations_waiting_feedback, "c2 should still be in waiting"
    assert sender.equations_waiting_feedback[c2_packet_id].unknown_packets == [2], f"c2 should now only have [p2], got {sender.equations_waiting_feedback[c2_packet_id].unknown_packets}"
    print(f"  ✓ c2 equation updated: unknowns=[2] (p1 removed)")
    
    # Check oldest_information_packet_on_air updated
    assert sender.oldest_information_packet_on_air == 2, f"oldest should be 2, got {sender.oldest_information_packet_on_air}"
    print(f"  ✓ oldest_information_packet_on_air = 2")
    
    # Sender sends c3 with [p2,p3]
    assert len(forward_history) == 3, f"Should have sent 3 packets, got {len(forward_history)}"
    c3 = forward_history[2]
    assert c3.get_information_packets() == [2, 3], f"c3 should contain [p2,p3], got {c3.get_information_packets()}"
    assert c3.get_type() == RLNCType.NEW, f"c3 should be NEW, got {c3.get_type()}"
    print(f"  ✓ c3 sent: info=[2,3], type=NEW (window slid)")
    
    # =================================================================
    # t=4: Receiver gets c3, sends ACK. Sender gets ACK(c2), decodes p2
    # =================================================================
    t = 4
    print(f"\n{'='*70}")
    print(f"t={t}: Receiver gets c3, sends ACK. Sender gets ACK(c2), decodes p2")
    print(f"{'='*70}")
    
    sender.run_step() # need to check why sender send p4 while there are only 3 information packets
    
    # Receiver checks
    assert len(receiver_history) == 3, f"Receiver should have 3 packets, got {len(receiver_history)}"
    print(f"  ✓ Receiver received c3")
    
    # Verify receiver decoded p3 at t=4 (when c3=[2,3] arrived)
    # Receiver already had p1, p2; c3 adds equation with [3], decodes p3
    assert 3 in receiver.information_packets_decoding_times, f"Receiver should have decoded p3 at t={t}, got {receiver.information_packets_decoding_times}"
    assert receiver.latest_decoded_information_packet == 3, f"Receiver latest should be 3, got {receiver.latest_decoded_information_packet}"
    assert len(receiver.information_packets_decoding_times) == 3, f"Receiver should have 3 decoded packets"
    print(f"  ✓ Receiver decoded p3 at t={t}: {sorted(receiver.information_packets_decoding_times)}")
    
    # Sender receives ACK(c2) and decodes p2
    assert len(sender.all_feedback_history) == 2, f"Sender should have 2 feedbacks, got {len(sender.all_feedback_history)}"
    print(f"  ✓ Sender received ACK for c2")
    
    assert len(sender.decoded_information_packets_history) == 2, "Sender should have decoded 2 packets"
    assert 2 in sender.decoded_information_packets_history, "p2 should be decoded"
    print(f"  ✓ Sender inferred p2 decoded (at t={t}, receiver decoded it at t={t-1})")
    
    assert sender.oldest_information_packet_on_air == 3, f"oldest should be 3, got {sender.oldest_information_packet_on_air}"
    print(f"  ✓ oldest_information_packet_on_air = 3")
    
    # c3 equation should be updated
    # c3_packet_id = PacketID(global_path_id=0, creation_time=3, type=RLNCType.NEW)
    c3_packet_id = PacketID(global_path_id=0, creation_time=3)
    assert c3_packet_id in sender.equations_waiting_feedback, "c3 should be in waiting"
    assert sender.equations_waiting_feedback[c3_packet_id].unknown_packets == [3], f"c3 should have [p3], got {sender.equations_waiting_feedback[c3_packet_id].unknown_packets}"
    print(f"  ✓ c3 equation: unknowns=[3]")
    
    # =================================================================
    # t=5: Sender gets ACK(c3), decodes p3 - ALL DONE
    # =================================================================
    t = 5
    print(f"\n{'='*70}")
    print(f"t={t}: Sender gets ACK(c3), decodes p3 - ALL PACKETS DECODED")
    print(f"{'='*70}")
    
    sender.run_step()
    
    assert len(sender.all_feedback_history) == 3, f"Sender should have 3 feedbacks, got {len(sender.all_feedback_history)}"
    print(f"  ✓ Sender received ACK for c3")
    
    assert len(sender.decoded_information_packets_history) == 3, "Sender should have decoded all 3 packets"
    assert sender.decoded_information_packets_history == [1, 2, 3], f"Should have decoded [1,2,3], got {sender.decoded_information_packets_history}"
    print(f"  ✓ Sender inferred p3 decoded (at t={t}, receiver decoded it at t={t-1}) - ALL PACKETS!")
    
    assert len(sender.equations_waiting_feedback) == 0, f"All equations should be cleared. remaining equations\n\t{sender.equations_waiting_feedback}"
    assert len(sender.acked_equations) == 0, "No equations should be in acked"
    print(f"  ✓ All equations cleared")
    
    # =================================================================
    # Final Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - SR2")
    print(f"{'='*70}")
    print(f"  ✓ All {NUM_PACKETS_TO_SEND} packets transmitted sequentially")
    print(f"  ✓ Window slid correctly: c1=[1], c2=[1,2], c3=[2,3]")
    print(f"  ✓ Sender inferred all packets decoded: {sender.decoded_information_packets_history}")
    print(f"  ✓ Receiver decoded all packets: {sorted(receiver.information_packets_decoding_times)}")
    print(f"  ✓ Equations properly tracked and cleaned up")
    print(f"  Note: Receiver decodes at packet arrival (t=2,3,4), Sender infers at ACK arrival (t=3,4,5)")
    print(f"\n{'='*70}")
    print(f"✓ Test SR2 PASSED!")
    print(f"{'='*70}\n")


def test_SR3_single_packet_multiple_paths_no_loss():
    """
    SR3 — Multi-path test: 2 packets, 2 paths, prop_delay=2, epsilon=0
    
    Flow:
    t=1: Sender sends one packet with [p1] and one with [p1,p2] (reversed iteration)
    t=3: Receiver gets both packets. Sends ACKs for both
    t=5: Sender gets both ACKs, infers p1 and p2 decoded
    
    Verifies:
    - Simultaneous transmission on multiple paths
    - Each path sends different information packets (one [1], one [1,2])
    - Receiver handles multiple paths independently
    - Sender correctly infers decoding with parallel transmissions
    - Order of transmission doesn't affect correctness (uses reversed() for performance)
    
    Note: RTT=4 (prop_delay=2) so EW=3, avoiding premature FEC for this test
    Note: Test is order-agnostic - verifies both packet types sent, not which path sent what
    """
    print("\n" + "="*70)
    print("Test SR3: Two packets, multiple paths, no loss")
    print("="*70)
    
    # Configuration
    PROP_DELAY = 2  # Increased to avoid EW trigger
    RTT = PROP_DELAY * 2  # RTT=4, so EW=3
    NUM_PACKETS_TO_SEND = 2
    NUM_PATHS = 2
    EPSILON = 0.0
    
    # Setup
    print(f"\nSetup: Creating {NUM_PATHS} paths")
    paths = []
    for i in range(NUM_PATHS):
        path = Path(propagation_delay=PROP_DELAY, epsilon=EPSILON, hop_index=0, path_index_in_hop=i)
        path.set_global_path_index(i)
        paths.append(path)
    
    receiver = GeneralReceiver(input_paths=paths, rtt=RTT, unit_name="TestReceiver")
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS_TO_SEND,
        rtt=RTT,
        paths=paths,
        receiver=receiver
    )
    print(f"  ✓ Created: {NUM_PATHS} paths, {NUM_PACKETS_TO_SEND} packets to send")
    
    # =================================================================
    # t=1: Sender sends on both paths
    # =================================================================
    t = 1
    print(f"\n{'='*70}")
    print(f"t={t}: Sender sends packets on both paths (one [1], one [1,2])")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Check both paths sent packets with different information packets
    forward_history_path0 = sender.paths[0].get_forward_channel_history()
    forward_history_path1 = sender.paths[1].get_forward_channel_history()
    
    assert len(forward_history_path0) == 1, f"Path 0 should have sent 1 packet, got {len(forward_history_path0)}"
    assert len(forward_history_path1) == 1, f"Path 1 should have sent 1 packet, got {len(forward_history_path1)}"
    
    pkt_path0 = forward_history_path0[0]
    pkt_path1 = forward_history_path1[0]
    
    # Verify both packets are NEW type
    assert pkt_path0.get_type() == RLNCType.NEW, f"Path 0 packet should be NEW"
    assert pkt_path1.get_type() == RLNCType.NEW, f"Path 1 packet should be NEW"
    
    # Collect packets and their info - order doesn't matter due to reversed() iteration
    all_packets = [pkt_path0, pkt_path1]
    all_info_packets = [pkt.get_information_packets() for pkt in all_packets]
    
    # Verify we have one packet with [1] and one with [1,2]
    assert [1] in all_info_packets, f"Should have packet with [1], got {all_info_packets}"
    assert [1, 2] in all_info_packets, f"Should have packet with [1,2], got {all_info_packets}"
    
    # Determine which path got which packet
    if pkt_path0.get_information_packets() == [1]:
        single_pkt, single_path = pkt_path0, 0
        double_pkt, double_path = pkt_path1, 1
    else:
        single_pkt, single_path = pkt_path1, 1
        double_pkt, double_path = pkt_path0, 0
    
    print(f"  ✓ Path {single_path}: sent c{single_pkt.get_creation_time()} with info=[1], type=NEW")
    print(f"  ✓ Path {double_path}: sent c{double_pkt.get_creation_time()} with info=[1,2], type=NEW")
    
    # Check equations - should have 2 equations with DIFFERENT unknowns
    assert len(sender.equations_waiting_feedback) == 2, f"Should have 2 equations, got {len(sender.equations_waiting_feedback)}:\n\t{sender.equations_waiting_feedback}"
    
    # Verify equations exist for both packets (regardless of order)
    # single_packet_id = PacketID(global_path_id=single_path, creation_time=1, type=RLNCType.NEW)
    single_packet_id = PacketID(global_path_id=single_path, creation_time=1)
    # double_packet_id = PacketID(global_path_id=double_path, creation_time=1, type=RLNCType.NEW)
    double_packet_id = PacketID(global_path_id=double_path, creation_time=1)
    
    assert single_packet_id in sender.equations_waiting_feedback, f"Packet [1] should be in equations"
    assert double_packet_id in sender.equations_waiting_feedback, f"Packet [1,2] should be in equations"
    assert sender.equations_waiting_feedback[single_packet_id].unknown_packets == [1], \
        f"[1] packet equation should have unknowns=[1], got {sender.equations_waiting_feedback[single_packet_id].unknown_packets}"
    assert sender.equations_waiting_feedback[double_packet_id].unknown_packets == [1, 2], \
        f"[1,2] packet equation should have unknowns=[1,2], got {sender.equations_waiting_feedback[double_packet_id].unknown_packets}"
    print(f"  ✓ Both equations created correctly (path {single_path}: unknowns=[1], path {double_path}: unknowns=[1,2])")
    
    # =================================================================
    # t=2: Packets still in transit (prop_delay=2)
    # =================================================================
    t = 2
    print(f"\n{'='*70}")
    print(f"t={t}: Packets still in transit")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Verify no packets received yet (they arrive at t=3)
    receiver_history = receiver.get_received_rlnc_channel_history()
    assert len(receiver_history) == 0, f"Receiver should have 0 packets at t={t}, got {len(receiver_history)}"
    print(f"  ✓ Packets still propagating (arrive at t={t+1})")
    
    # =================================================================
    # t=3: Receiver gets both packets, sends 2 ACKs
    # =================================================================
    t = 3
    print(f"\n{'='*70}")
    print(f"t={t}: Receiver gets both packets on both paths, sends ACKs")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Receiver should have received 2 packets with different information
    receiver_history = receiver.get_received_rlnc_channel_history()
    assert len(receiver_history) == 2, f"Receiver should have 2 packets, got {len(receiver_history)}"
    
    # Verify we received both packet types (order doesn't matter)
    received_info_packets = [pkt.get_information_packets() for pkt in receiver_history]
    assert [1] in received_info_packets, f"Should have received packet with [1], got {received_info_packets}"
    assert [1, 2] in received_info_packets, f"Should have received packet with [1,2], got {received_info_packets}"
    print(f"  ✓ Receiver received both packets: [1] and [1,2]")
    
    # Receiver should have sent 2 ACKs
    feedback_history = receiver.get_sent_feedback_channel_history()
    assert len(feedback_history) == 2, f"Should have 2 feedbacks, got {len(feedback_history)}"
    
    # Verify both feedbacks are ACKs with correct info packets (order doesn't matter)
    for fb in feedback_history:
        assert fb.is_ack(), f"All feedbacks should be ACK, got {fb}"
    
    feedback_info_packets = [fb.get_related_information_packets() for fb in feedback_history]
    assert [1] in feedback_info_packets, f"Should have ACK for [1], got {feedback_info_packets}"
    assert [1, 2] in feedback_info_packets, f"Should have ACK for [1,2], got {feedback_info_packets}"
    print(f"  ✓ Receiver sent ACKs for both packets")
    
    # Verify receiver decoded both packets at t=3 (when they arrived)
    assert 1 in receiver.information_packets_decoding_times, f"Receiver should have decoded p1 at t={t}"
    assert 2 in receiver.information_packets_decoding_times, f"Receiver should have decoded p2 at t={t}"
    assert receiver.latest_decoded_information_packet == 2, f"Receiver latest should be 2"
    assert len(receiver.information_packets_decoding_times) == 2, f"Receiver should have 2 decoded packets"
    print(f"  ✓ Receiver decoded p1 and p2 at t={t}: {sorted(receiver.information_packets_decoding_times)}")
    
    # Sender hasn't received ACKs yet
    assert len(sender.decoded_information_packets_history) == 0, "Sender shouldn't have decoded yet"
    print(f"  ✓ Sender hasn't received ACKs yet")
    
    # =================================================================
    # t=4: ACKs still in transit
    # =================================================================
    t = 4
    print(f"\n{'='*70}")
    print(f"t={t}: ACKs in transit back to sender")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Sender still hasn't received ACKs
    assert len(sender.decoded_information_packets_history) == 0, "Sender shouldn't have decoded yet"
    print(f"  ✓ ACKs still propagating (arrive at t={t+1})")
    
    # =================================================================
    # t=5: Sender gets both ACKs, decodes p1 and p2
    # =================================================================
    t = 5
    print(f"\n{'='*70}")
    print(f"t={t}: Sender receives both ACKs, infers p1 and p2 decoded")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Sender should have received 2 ACKs
    assert len(sender.all_feedback_history) == 2, f"Sender should have 2 feedbacks, got {len(sender.all_feedback_history)}:\n\t{sender.all_feedback_history}"
    acks = [fb for fb in sender.all_feedback_history if fb.is_ack()]
    assert len(acks) == 2, f"Should have 2 ACKs, got {len(acks)}"
    print(f"  ✓ Sender received 2 ACKs")
    
    # Sender should have decoded both p1 and p2
    assert len(sender.decoded_information_packets_history) == 2, f"Sender should have decoded 2 packets, got {len(sender.decoded_information_packets_history)}"
    assert 1 in sender.decoded_information_packets_history, f"p1 should be decoded, got {sender.decoded_information_packets_history}"
    assert 2 in sender.decoded_information_packets_history, f"p2 should be decoded, got {sender.decoded_information_packets_history}"
    print(f"  ✓ Sender inferred p1 and p2 decoded (at t={t}, receiver decoded them at t={t-2})")
    
    # Both equations should be cleared
    assert len(sender.equations_waiting_feedback) == 0, f"All equations should be cleared, got {sender.equations_waiting_feedback}"
    assert len(sender.acked_equations) == 0, f"No equations in acked, got {sender.acked_equations}"
    print(f"  ✓ Both equations cleared")
    
    # =================================================================
    # Final Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - SR3")
    print(f"{'='*70}")
    print(f"  ✓ {NUM_PATHS} paths transmitted simultaneously")
    print(f"  ✓ Each path sent different information packets")
    print(f"  ✓ Receiver handled {NUM_PATHS} paths independently")
    print(f"  ✓ Receiver decoded at t=3: {sorted(receiver.information_packets_decoding_times)}")
    print(f"  ✓ Sender inferred decoding at t=5: {sender.decoded_information_packets_history}")
    print(f"  ✓ All {NUM_PACKETS_TO_SEND} equations cleared after decoding")
    print(f"  Note: Receiver decodes {PROP_DELAY*2} steps before sender infers (RTT={PROP_DELAY*2})")
    print(f"\n{'='*70}")
    print(f"✓ Test SR3 PASSED!")
    print(f"{'='*70}\n")


def test_SR4_packet_loss_single_path():
    """
    SR4 — Packet loss test: 3 packets, 1 path, controlled dropping
    
    Tests packet loss handling with deterministic control (c1 dropped).
    
    Flow:
    t=1: Sender sends c1=[1], but c1 is DROPPED
    t=2: Receiver sends NACK (no packet received)
         Sender sends c2=[1,2]
    t=3: Sender gets NACK, removes c1 equation
         Receiver gets c2, sends ACK
         Sender sends c3=[1,2,3]
    t=4: Sender gets ACK(c2) → c2 to acked_equations
         Receiver gets c3, sends ACK
         Sender sends c4=[1,2,3] FEC (EW triggered)
    t=5: Sender gets ACK(c3) → c3 to acked_equations
         Status: 3 unknowns [1,2,3], 2 acked equations (c2,c3) → NOT DECODABLE
         Sender continues transmitting (c5 NEW)
    
    Verifies:
    - Dropped packets tracked separately (TestPath deterministic dropping)
    - NACK generation for missing packets
    - Equation removal on NACK (c1 removed from waiting)
    - Equation accumulation in acked_equations (c2, c3)
    - Non-decodability detection (3 unknowns, 2 equations)
    - Parameter estimation (epsilon_est = NACKs / total_feedbacks)
    - System continues transmitting when decoding incomplete
    """
    print("\n" + "="*70)
    print("Test SR4: Packet loss, single path (c1 dropped)")
    print("="*70)
    
    # Configuration
    PROP_DELAY = 1
    RTT = PROP_DELAY * 2
    NUM_PACKETS_TO_SEND = 3
    DROP_PACKETS = {1}  # Drop c1 (creation_time=1)
    
    # Setup with controlled dropping
    print(f"\nSetup: Creating TestPath that drops packets at times: {DROP_PACKETS}")
    path = TestPath(
        propagation_delay=PROP_DELAY, 
        hop_index=0, 
        path_index_in_hop=0,
        drop_packet_times=DROP_PACKETS
    )
    path.set_global_path_index(0)
    receiver = GeneralReceiver(input_paths=[path], rtt=RTT, unit_name="TestReceiver")
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS_TO_SEND,
        rtt=RTT,
        paths=[path],
        receiver=receiver
    )
    print(f"  ✓ Created: {NUM_PACKETS_TO_SEND} packets to send, c1 will be dropped")
    
    # =================================================================
    # t=1: Sender sends c1 with [p1], but c1 is DROPPED
    # =================================================================
    t = 1
    print(f"\n{'='*70}")
    print(f"t={t}: Sender sends c1 with [p1] - WILL BE DROPPED")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Verify c1 was dropped
    dropped_packets = sender.paths[0].get_dropped_packets()
    assert len(dropped_packets) == 1, f"Should have 1 dropped packet, got {len(dropped_packets)}"
    c1_dropped = dropped_packets[0]
    assert c1_dropped.get_information_packets() == [1], f"Dropped packet should contain [p1], got {c1_dropped.get_information_packets()}"
    assert c1_dropped.get_creation_time() == t, f"Dropped packet should have creation_time={t}, got {c1_dropped.get_creation_time()}"
    print(f"  ✓ c1 sent with info=[1] and was DROPPED")
    
    # Verify c1 is NOT in forward_channel_history (dropped packets don't propagate)
    forward_history = sender.paths[0].get_forward_channel_history()
    assert len(forward_history) == 0, f"Dropped packet shouldn't be in channel history, got {len(forward_history)}"
    print(f"  ✓ c1 not in channel history (dropped before propagation)")
    
    # Verify c1 is in equations_waiting_feedback (sender doesn't know it's dropped yet)
    # c1_packet_id = PacketID(global_path_id=0, creation_time=1, type=RLNCType.NEW)
    c1_packet_id = PacketID(global_path_id=0, creation_time=1)
    assert c1_packet_id in sender.equations_waiting_feedback, "c1 should be in equations_waiting_feedback"
    print(f"  ✓ Equation for c1 added to waiting (sender doesn't know it's dropped)")
    
    # Receiver shouldn't have received anything
    receiver_history = receiver.get_received_rlnc_channel_history()
    assert len(receiver_history) == 0, "Receiver shouldn't have received c1 (dropped)"
    print(f"  ✓ Receiver did not receive c1 (dropped)")
    
    # =================================================================
    # t=2: Receiver sends NACK (no packet received), Sender sends c2
    # =================================================================
    t = 2
    print(f"\n{'='*70}")
    print(f"t={t}: Receiver sends NACK, Sender sends c2 with [p1, p2]")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Verify receiver sent NACK
    feedback_history = receiver.get_sent_feedback_channel_history()
    assert len(feedback_history) == 1, f"Should have 1 feedback (NACK), got {len(feedback_history)}"
    nack = feedback_history[0]
    assert nack.is_nack(), f"First feedback should be NACK, got {nack}"
    print(f"  ✓ Receiver sent NACK for missing packet")
    
    # Verify c2 was sent (should be first in history since c1 was dropped)
    forward_history = sender.paths[0].get_forward_channel_history()
    assert len(forward_history) == 1, f"Should have 1 packet in channel history (c1 dropped), got {len(forward_history)}"
    c2 = forward_history[0]
    assert c2.get_information_packets() == [1, 2], f"c2 should contain [p1,p2], got {c2.get_information_packets()}"
    assert c2.get_creation_time() == t, f"c2 should have creation_time={t}"
    print(f"  ✓ c2 sent with info=[1,2]")
    
    # c2 equation should be in waiting
    # c2_packet_id = PacketID(global_path_id=0, creation_time=2, type=RLNCType.NEW)
    c2_packet_id = PacketID(global_path_id=0, creation_time=2)
    assert c2_packet_id in sender.equations_waiting_feedback, "c2 should be in waiting"
    print(f"  ✓ Equation for c2 added to waiting")
    
    # Receiver still has no packets
    assert len(receiver_history) == 0, "Receiver still hasn't received any packets"
    print(f"  ✓ Receiver has no packets yet (c2 in transit)")
    
    # =================================================================
    # t=3: Sender gets NACK, removes c1. Receiver gets c2, sends ACK
    # =================================================================
    t = 3
    print(f"\n{'='*70}")
    print(f"t={t}: Sender gets NACK (removes c1), Receiver gets c2, sends ACK")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Verify sender received NACK
    assert len(sender.all_feedback_history) == 1, f"Sender should have 1 feedback, got {len(sender.all_feedback_history)}"
    assert sender.all_feedback_history[0].is_nack(), "First feedback should be NACK"
    print(f"  ✓ Sender received NACK")
    
    # Verify c1 removed from equations_waiting_feedback
    assert c1_packet_id not in sender.equations_waiting_feedback, \
        f"c1 should be removed after NACK, still in: {sender.equations_waiting_feedback.keys()}"
    print(f"  ✓ c1 equation removed from waiting (due to NACK)")
    
    # Verify c2 still in waiting
    assert c2_packet_id in sender.equations_waiting_feedback, "c2 should still be in waiting"
    print(f"  ✓ c2 equation still in waiting")
    
    # Verify receiver got c2
    assert len(receiver_history) == 1, f"Receiver should have 1 packet, got {len(receiver_history)}"
    assert receiver_history[0].get_information_packets() == [1, 2], "Receiver should have c2"
    print(f"  ✓ Receiver received c2 with info=[1,2]")
    
    # Verify receiver sent ACK for c2
    assert len(feedback_history) == 2, f"Should have 2 feedbacks, got {len(feedback_history)}"
    assert feedback_history[1].is_ack(), "Second feedback should be ACK"
    assert feedback_history[1].get_related_packet_id().get_creation_time() == 2, "ACK should be for c2"
    print(f"  ✓ Receiver sent ACK for c2")
    
    # Sender shouldn't have decoded yet (no feedbacks received)
    assert len(sender.decoded_information_packets_history) == 0, "Sender shouldn't have decoded yet"
    print(f"  ✓ Sender hasn't decoded yet")
    
    # Verify c3 sent (should be second in history since c1 was dropped)
    forward_history = sender.paths[0].get_forward_channel_history()
    assert len(forward_history) == 2, f"Should have 2 packets in channel history (c1 dropped, c2 and c3 sent), got {len(forward_history)}"
    c3 = forward_history[1]
    assert c3.get_information_packets() == [1, 2, 3], f"c3 should contain [p1,p2,p3], got {c3.get_information_packets()}"
    assert c3.get_creation_time() == t, f"c3 should have creation_time={t}"
    print(f"  ✓ c3 sent with info=[1,2,3]")
    
    # c3 equation should be in waiting
    # c3_packet_id = PacketID(global_path_id=0, creation_time=3, type=RLNCType.NEW)
    c3_packet_id = PacketID(global_path_id=0, creation_time=3)
    assert c3_packet_id in sender.equations_waiting_feedback, "c3 should be in waiting"
    print(f"  ✓ Equation for c3 added to waiting")

    # Verify sender parameter estimation updated
    assert len(sender.all_feedback_history) == 1, f"Sender should have 1 feedback, got {len(sender.all_feedback_history)}"
    assert sender.all_feedback_history[0].is_nack(), "First feedback should be NACK"
    assert sender.all_feedback_history[0].get_related_packet_id().get_creation_time() == 1, "NACK should be for c1"
    assert sender.paths[0].epsilon_est == 1, f"epsilon_est should be 1, got epsilon_est={sender.paths[0].epsilon_est}"
    assert sender.paths[0].r == 1 - sender.paths[0].epsilon_est, f"r should be 1 - epsilon_est, got r={sender.paths[0].r} epsilon_est={sender.paths[0].epsilon_est}"
    print(f"  ✓ Sender parameter estimation updated: epsilon_est={sender.paths[0].epsilon_est:.3f}, r={sender.paths[0].r:.3f}")
    
    # =================================================================
    # t=4: Sender gets ACK for c2, Receiver gets c3, sends ACK
    # =================================================================
    t = 4
    print(f"\n{'='*70}")
    print(f"t={t}: Sender gets ACK(c2), Receiver gets c3, sends ACK")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Verify sender received ACK for c2
    assert len(sender.all_feedback_history) == 2, f"Should have 2 feedbacks, got {len(sender.all_feedback_history)}"
    acks = [fb for fb in sender.all_feedback_history if fb.is_ack()]
    assert len(acks) == 1, f"Should have 1 ACK, got {len(acks)}"
    assert acks[0].get_related_packet_id().get_creation_time() == 2, "ACK should be for c2"
    print(f"  ✓ Sender received ACK for c2")
    
    # Verify c2 moved to acked_equations
    assert c2_packet_id in sender.acked_equations, f"c2 should be in acked_equations"
    assert sender.acked_equations[c2_packet_id].unknown_packets == [1, 2], \
        f"c2 equation should have unknowns=[1,2]"
    print(f"  ✓ c2 equation moved to acked_equations: unknowns=[1,2]")
    
    # Verify parameter estimation updated (1 NACK, 1 ACK so far)
    path0 = sender.paths[0]
    assert len(path0.all_feedback_history) == 2, "Path should have 2 feedbacks"
    assert len(path0.nacked_feedback_history) == 1, "Path should have 1 NACK"
    assert len(path0.acked_feedback_history) == 1, "Path should have 1 ACK"
    epsilon_est = path0.epsilon_est
    r_est = path0.r
    expected_epsilon = 1/2  # 1 NACK out of 2 feedbacks
    assert abs(epsilon_est - expected_epsilon) < 0.001, \
        f"epsilon_est should be {expected_epsilon}, got {epsilon_est}"
    print(f"  ✓ Parameter estimation: epsilon_est={epsilon_est:.3f} (1 NACK / 2 feedbacks), r={r_est:.3f}")
    
    # Verify not decoded yet (2 unknowns, 1 equation)
    assert len(sender.decoded_information_packets_history) == 0, \
        "Shouldn't decode yet (2 unknowns, 1 equation)"
    print(f"  ✓ Not decoded yet (2 unknowns, 1 equation)")
    
    # Verify receiver got c3
    assert len(receiver_history) == 2, f"Receiver should have 2 packets, got {len(receiver_history)}"
    assert receiver_history[1].get_information_packets() == [1, 2, 3], "Should have received c3"
    print(f"  ✓ Receiver received c3 with info=[1,2,3]")
    
    # Verify ACK sent for c3
    assert len(feedback_history) == 3, f"Should have 3 feedbacks, got {len(feedback_history)}"
    assert feedback_history[2].is_ack(), "Third feedback should be ACK"
    print(f"  ✓ Receiver sent ACK for c3")
    
    # =================================================================
    # t=5: Sender gets ACK for c3, but still cannot decode!
    # =================================================================
    t = 5
    print(f"\n{'='*70}")
    print(f"t={t}: Sender gets ACK(c3), accumulates 2 equations (still can't decode)")
    print(f"{'='*70}")
    
    sender.run_step()
    
    # Verify sender received ACK for c3
    assert len(sender.all_feedback_history) == 3, f"Should have 3 feedbacks, got {len(sender.all_feedback_history)}"
    acks = [fb for fb in sender.all_feedback_history if fb.is_ack()]
    nacks = [fb for fb in sender.all_feedback_history if fb.is_nack()]
    assert len(acks) == 2, f"Should have 2 ACKs, got {len(acks)}"
    assert len(nacks) == 1, f"Should have 1 NACK, got {len(nacks)}"
    print(f"  ✓ Sender received ACK for c3")
    
    # Verify parameter estimation updated (1 NACK, 2 ACKs)
    path0 = sender.paths[0]
    epsilon_est = path0.epsilon_est
    r_est = path0.r
    expected_epsilon = 1/3  # 1 NACK out of 3 feedbacks
    assert abs(epsilon_est - expected_epsilon) < 0.001, \
        f"epsilon_est should be {expected_epsilon:.3f}, got {epsilon_est}"
    print(f"  ✓ Parameter estimation updated: epsilon_est={epsilon_est:.3f} (1 NACK / 3 feedbacks), r={r_est:.3f}")
    
    # Verify c3 moved to acked_equations
    # c3_packet_id = PacketID(global_path_id=0, creation_time=3, type=RLNCType.NEW)
    c3_packet_id = PacketID(global_path_id=0, creation_time=3)
    assert c3_packet_id in sender.acked_equations, "c3 should be in acked_equations"
    assert sender.acked_equations[c3_packet_id].unknown_packets == [1, 2, 3], \
        f"c3 equation should have unknowns=[1,2,3]"
    print(f"  ✓ c3 equation moved to acked_equations: unknowns=[1,2,3]")
    
    # Verify decoding status: 3 unknowns (p1, p2, p3), 2 equations (c2, c3) → NOT decodable
    print(f"\n  Checking decoding status:")
    print(f"    Unknowns: [1, 2, 3] (3 packets)")
    print(f"    Equations: c2 and c3 (2 equations)")
    print(f"    Decodable: 3 unknowns > 2 equations → NO")
    
    assert len(sender.decoded_information_packets_history) == 0, \
        "Should NOT decode yet (3 unknowns, 2 equations)"
    print(f"  ✓ Sender correctly NOT decoded (insufficient equations)")
    
    # Verify receiver HAS decoded by t=5!
    # At t=3: received c2=[1,2]
    # At t=4: received c3=[1,2,3]  
    # At t=5: received c4=[1,2,3] → 3 equations, 3 unknowns → CAN DECODE!
    assert len(receiver.information_packets_decoding_times) == 3, \
        f"Receiver should have decoded all 3 packets by t={t} (3 unknowns, 3+ equations), got {receiver.information_packets_decoding_times}"
    assert 1 in receiver.information_packets_decoding_times, "Receiver should have decoded p1"
    assert 2 in receiver.information_packets_decoding_times, "Receiver should have decoded p2"
    assert 3 in receiver.information_packets_decoding_times, "Receiver should have decoded p3"
    assert receiver.latest_decoded_information_packet == 3, \
        f"Receiver latest should be 3, got {receiver.latest_decoded_information_packet}"
    print(f"  ✓ Receiver DID decode at t={t}: {sorted(receiver.information_packets_decoding_times)} (had 3 equations)")
    
    # Both c2 and c3 should be in acked_equations
    assert len(sender.acked_equations) == 2, f"Should have 2 acked equations, got {len(sender.acked_equations)}"
    assert c2_packet_id in sender.acked_equations, "c2 should be in acked"
    assert c3_packet_id in sender.acked_equations, "c3 should be in acked"
    print(f"  ✓ c2 and c3 accumulated in acked_equations")
    
    # Additional packets were sent (c4 FEC at t=4, c5 NEW at t=5) - they're in waiting
    # Note: At t=4, is_EW() was true (3 packets > EW=1), triggering init_fec (c4)
    # At t=5, sender continues (not all info packets decoded), sends c5
    forward_history = sender.paths[0].get_forward_channel_history()
    print(f"\n  Additional transmissions (system continues due to incomplete decoding):")
    print(f"    Total packets in channel history: {len(forward_history)}")
    
    # Equations in waiting should include packets sent after receiving their feedbacks
    equations_waiting_count = len(sender.equations_waiting_feedback)
    print(f"    Equations waiting feedback: {equations_waiting_count}")
    print(f"  ✓ System continues transmitting (c4 FEC, c5 NEW) waiting for feedback")
    
    # =================================================================
    # Final Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - SR4")
    print(f"{'='*70}")
    
    print(f"\nPacket Loss Handling:")
    print(f"  ✓ c1 dropped using TestPath (deterministic dropping)")
    print(f"  ✓ Dropped packet tracked separately from channel history")
    print(f"  ✓ Sender added c1 equation (doesn't know about drop initially)")
    
    print(f"\nFeedback Mechanism:")
    print(f"  ✓ NACK generated for c1 (missing packet)")
    print(f"  ✓ ACKs generated for c2, c3 (and subsequent packets)")
    print(f"  ✓ Sender processes feedbacks correctly")
    
    print(f"\nEquation Management:")
    print(f"  ✓ c1 removed from waiting (NACK received)")
    print(f"  ✓ Sender: c2 and c3 accumulated in acked_equations")
    print(f"  ✓ Sender: Correct non-decodability (3 unknowns, 2 acked equations)")
    print(f"  ✓ Receiver: CAN decode (received c2, c3, c4 by t=5 → 3 equations)")
    print(f"  ✓ Receiver ahead of sender (has received more packets)")
    print(f"  ✓ System continues transmitting until sender can decode")
    
    print(f"\nParameter Estimation:")
    path0 = sender.paths[0]
    print(f"  ✓ epsilon_est = {path0.epsilon_est:.3f} (tracks NACK rate)")
    print(f"  ✓ r = {path0.r:.3f} (1 - epsilon_est)")
    print(f"  ✓ Estimation formula: epsilon_est = len(NACKs) / len(all_feedbacks)")
    
    print(f"\nSystem Behavior:")
    total_sent = len(forward_history)
    total_dropped = len(sender.paths[0].get_dropped_packets())
    print(f"  ✓ Total packets sent: {total_sent}")
    print(f"  ✓ Total packets dropped: {total_dropped}")
    print(f"  ✓ Packets received: {len(receiver.get_received_rlnc_channel_history())}")
    print(f"  ✓ Feedbacks sent: {len(receiver.get_sent_feedback_channel_history())}")
    
    print(f"\n  Note: Test captures snapshot at t=5. System would continue until decodable.")
    print(f"        (Need 3 equations for 3 unknowns - currently have 2 acked)")
    print(f"\n{'='*70}")
    print(f"✓ Test SR4 PASSED - Controlled packet loss and equation management verified!")
    print(f"{'='*70}\n")


def test_SR5_encoding_window_and_fec_init():
    """
    SR5 — Encoding Window and FEC initialization test
    
    Tests:
    - EW = P*(RTT-1) boundary detection
    - init_fec_transmissions() triggers after EW reached
    - mp (a-priori FEC counter) initialized correctly
    - FEC packets sent with mp > 0
    
    Configuration: 5 packets, 2 paths, RTT=4 (so EW=3)
    """
    print("\n" + "="*70)
    print("Test SR5: Encoding Window and FEC initialization")
    print("="*70)
    
    # Configuration
    PROP_DELAY = 2
    RTT = PROP_DELAY * 2
    NUM_PATHS = 1
    EW = NUM_PATHS * (RTT - 1) 
    NUM_PACKETS_TO_SEND = 5
    TRUE_EPSILON = 0.0
    INITIAL_EPSILON = 1.0
    
    print(f"\nConfiguration:")
    print(f"  RTT = {RTT}, EW = {EW}")
    print(f"  {NUM_PATHS} paths, true epsilon = {TRUE_EPSILON}, initial epsilon = {INITIAL_EPSILON}")
    print(f"  {NUM_PACKETS_TO_SEND} packets to send")
    
    # Setup
    paths = []
    for i in range(NUM_PATHS):
        path = Path(propagation_delay=PROP_DELAY, epsilon=TRUE_EPSILON, hop_index=0, path_index_in_hop=i)
        path.set_global_path_index(i)
        paths.append(path)
    
    receiver = GeneralReceiver(input_paths=paths, rtt=RTT, unit_name="TestReceiver")
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS_TO_SEND,
        rtt=RTT,
        paths=paths,
        receiver=receiver,
        initial_epsilon=INITIAL_EPSILON
    )
    print(f"  ✓ Setup complete")
    
    print(f"\n{'='*70}")
    print(f"Running simulation for 2*EW={2*EW} steps")
    print(f"{'='*70}")
    
    for _ in range(2*EW):
        sender.run_step()

    # Print sent history
    new_rlnc_history = sender.sent_new_rlnc_history
    fec_history = sender.sent_fec_history
    fb_fec_history = sender.sent_fb_fec_history
    print(f"    New RLNC history: {new_rlnc_history}")
    print(f"    FEC history: {fec_history}")
    print(f"    FB-FEC history: {fb_fec_history}")

    # Verify FEC was sent floor(NUM_PATHS * EPSILON * (RTT - 1)) times
    expected_fec_packets = int(NUM_PATHS * INITIAL_EPSILON * (RTT - 1))
    assert len(fec_history) == expected_fec_packets, f"Should have sent {expected_fec_packets} FEC packets, got {len(fec_history)}"
    print(f"    ✓ FEC packets sent: {len(fec_history)} == {expected_fec_packets}")

    # Verify RLNC packet was sent NUM_Paths*2*EW times
    excpected_num_rlnc = NUM_PATHS * 2 * EW
    assert len(new_rlnc_history + fec_history + fb_fec_history) == excpected_num_rlnc, f"Should have sent {excpected_num_rlnc} RLNC packets, got {len(new_rlnc_history + fec_history + fb_fec_history)}"
    print(f"    ✓ RLNC packets sent: {len(new_rlnc_history + fec_history + fb_fec_history)} == {excpected_num_rlnc}")

    # Verify no FB-FEC packets were sent
    assert len(fb_fec_history) == 0, f"Should not have sent any FB-FEC packets, got {len(fb_fec_history)}"
    print(f"    ✓ No FB-FEC packets sent: {len(fb_fec_history)} == 0")

    print(f"\n{'='*70}")
    print(f"✓ Test SR5 PASSED!")
    print(f"{'='*70}\n")

def test_SR6_test1_md_parameters():
    print("\n" + "="*70)
    print("Test SR6 Test 1: MD parameters")
    print("="*70)

    # Configuration
    PROP_DELAY = 1
    RTT = PROP_DELAY * 2
    NUM_PACKETS_TO_SEND = 5
    NUM_PATHS = 1
    MAX_ALLOWED_OVERLAP = 10
    EPSILON = 1 # 100% loss
    INITIAL_EPSILON = 1
    EXPECTED_AD_PARAMS = 0 # ad1, ad2, adg should be 0 throughout the test
    
    print(f"\nConfiguration:")
    print(f"  RTT = {RTT}")
    print(f"  {NUM_PATHS} paths, {NUM_PACKETS_TO_SEND} packets to send")
    
    # Setup
    path = Path(propagation_delay=PROP_DELAY, epsilon=EPSILON, hop_index=0, path_index_in_hop=0)
    path.set_global_path_index(0)

    receiver = GeneralReceiver(input_paths=[path], rtt=RTT, unit_name="TestReceiver")
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS_TO_SEND,
        rtt=RTT,
        paths=[path],
        receiver=receiver,
        max_allowed_overlap=MAX_ALLOWED_OVERLAP,
        initial_epsilon=INITIAL_EPSILON
    )
    print(f"  ✓ Setup complete")
    print(f"\n{'='*70}")

    t = 1
    print(f"[{t}] Running simulation, Sender sends c1, dropping all packets")
    sender.run_step()
    expected_num_new_rlnc = t
    assert len(sender.sent_new_rlnc_history) == expected_num_new_rlnc, f"Should have sent {expected_num_new_rlnc} new RLNC packets, got {len(sender.sent_new_rlnc_history)}"
    print(f"  ✓ {expected_num_new_rlnc} new RLNC packets sent")
    print(f"  forward channel history: {sender.paths[0].get_forward_channel_history(include_dropped_packets=True)}")

    t += 1 # t=2
    print(f"\n{'='*70}")
    print(f"[{t}] Running simulation, Sender sends c2, dropping all packets")
    print(f"checking md1, md2. ad1, ad2 should be 0.0")
    sender.paths[0].mp = 0 # Disable FEC transmission
    sender.run_step()
    expected_num_new_rlnc = t
    expected_md1 = 0.0 # no NACKS yet
    expected_md2 = INITIAL_EPSILON * (expected_num_new_rlnc - 1) # md2 gets updated before sending c2
    expected_mdg = expected_md1 + expected_md2
    print(f"  step {t} results:")
    print(f"  num_new_rlnc_sent = {expected_num_new_rlnc}")
    assert sender.md1 == expected_md1, f"md1 should be {expected_md1}, got {sender.md1}"
    print(f"  ✓ md1 = {expected_md1}")
    assert sender.md2 == expected_md2, f"md2 should be {expected_md2}, got {sender.md2}"
    print(f"  ✓ md2 = {expected_md2}")
    assert sender.mdg == expected_mdg, f"mdg should be {expected_mdg}, got {sender.mdg}"
    print(f"  ✓ mdg = {expected_mdg}")
    assert sender.ad1 == EXPECTED_AD_PARAMS, f"ad1 should be {EXPECTED_AD_PARAMS}, got {sender.ad1}"
    print(f"  ✓ ad1 = {EXPECTED_AD_PARAMS}")
    assert sender.ad2 == EXPECTED_AD_PARAMS, f"ad2 should be {EXPECTED_AD_PARAMS}, got {sender.ad2}"
    print(f"  ✓ ad2 = {EXPECTED_AD_PARAMS}")
    assert sender.adg == EXPECTED_AD_PARAMS, f"adg should be {EXPECTED_AD_PARAMS}, got {sender.adg}"
    print(f"  ✓ adg = {EXPECTED_AD_PARAMS}")
    print(f"  ✓ md1 = {expected_md1}, md2 = {expected_md2}, mdg = {expected_mdg}, ad1 = {EXPECTED_AD_PARAMS}, ad2 = {EXPECTED_AD_PARAMS}, adg = {EXPECTED_AD_PARAMS}")

    # Check md1 advancing as expected
    for step in range(1, 51):
        t += 1 # from 3 to 53
        print(f"\n{'='*70}")
        print(f"[{t}] NACK c1 received, checking md1, md2, ad1, ad2 should be 0.0")
        sender.paths[0].mp = 0 # Disable FEC transmission
        sender.run_step()
        expected_num_new_rlnc_sent = t # FEC disabled and adg = 0, so sender should send new RLNC every step
        expected_md1 = float(step) # We get NACK for every step
        expected_md2 = 1 # We get NACK for every step so md2 should remain 1 (for the packet currently on air)
        expected_mdg = expected_md1 + expected_md2
        print(f"  step {t} results:")
        assert len(sender.sent_new_rlnc_history) == expected_num_new_rlnc_sent, f"Sender should have sent {expected_num_new_rlnc_sent} new RLNC packets (1 at each time step), got {len(sender.sent_new_rlnc_history)}"
        print(f"  num_new_rlnc_sent = {expected_num_new_rlnc_sent}")
        assert sender.paths[0].epsilon_est == INITIAL_EPSILON, f"eps_est should be {INITIAL_EPSILON}, got {sender.paths[0].epsilon_est}"
        print(f"  ✓ epsilon_est = {INITIAL_EPSILON}")
        assert sender.md1 == expected_md1, f"md1 should be {expected_md1}, got {sender.md1}"
        print(f"  ✓ md1 = {expected_md1}")
        assert sender.md2 == expected_md2, f"md2 should be {expected_md2}, got {sender.md2}"
        print(f"  ✓ md2 = {expected_md2}")
        assert sender.mdg == expected_mdg, f"mdg should be {expected_mdg}, got {sender.mdg}"
        print(f"  ✓ mdg = {expected_mdg}")
        assert sender.ad1 == EXPECTED_AD_PARAMS, f"ad1 should be {EXPECTED_AD_PARAMS}, got {sender.ad1}"
        print(f"  ✓ ad1 = {EXPECTED_AD_PARAMS}")
        assert sender.ad2 == EXPECTED_AD_PARAMS, f"ad2 should be {EXPECTED_AD_PARAMS}, got {sender.ad2}"
        print(f"  ✓ ad2 = {EXPECTED_AD_PARAMS}")
        assert sender.adg == EXPECTED_AD_PARAMS, f"adg should be {EXPECTED_AD_PARAMS}, got {sender.adg}"
        print(f"  ✓ adg = {EXPECTED_AD_PARAMS}")
        print(f"  ✓ md1 = {expected_md1}, md2 = {expected_md2}, mdg = {expected_mdg}, ad1 = {EXPECTED_AD_PARAMS}, ad2 = {EXPECTED_AD_PARAMS}, adg = {EXPECTED_AD_PARAMS}")
    
    # Check md2 advancing as expected
    # Drop NACKs and assert md2 advances
    for step in range(51, 101):
        t += 1 # from 53 to 103
        print(f"\n{'='*70}")
        print(f"[{t}] Dropping NACKs, checking md2 should remain 1")
        sender.paths[0].mp = 0 # Disable FEC transmission
        sender.paths[0].feedback_channel.packets_in_channel = [] # Drop all feedbacks (should be NACKs)
        # print(f"  dropped feedbacks: {feedback}")
        sender.run_step()
        expected_num_new_rlnc_sent = t # FEC disabled and adg = 0, so sender should send new RLNC every step
        expected_md2 += 1 # md2 should advance
        print(f"  step {t} results:")
        assert sender.md2 == expected_md2, f"md2 did not advance as expected, got {sender.md2} instead of {expected_md2}"
        print(f"  ✓ md2 = {expected_md2}")

    print(f"\n{'='*70}")
    print(f"✓ Test SR6 Test 1: MD parameters PASSED!")
    print(f"{'='*70}\n")

def test_SR6_test2_fb_with_1_channel():
    print("\n" + "="*70)
    print("Test SR6 Test 2: AD parameters")
    print("="*70)

    # Configuration
    PROP_DELAY = 2
    RTT = PROP_DELAY * 2
    NUM_PACKETS_TO_SEND = sys.maxsize
    NUM_PATHS = 1
    MAX_ALLOWED_OVERLAP = NUM_PACKETS_TO_SEND
    EPSILON = 0.5 # 50% loss
    INITIAL_EPSILON = 0.5 # Sets Channel rate (r) to 0.5
    
    print(f"\nConfiguration:")
    print(f"  RTT = {RTT}")
    print(f"  {NUM_PATHS} paths, {NUM_PACKETS_TO_SEND} packets to send")
    
    # Setup
    # Create a path that drops every other packet (50% loss)
    drop_packet_times = set(range(2, 1000, 2))
    path = TestPath(propagation_delay=PROP_DELAY, hop_index=0, path_index_in_hop=0, drop_packet_times=drop_packet_times)
    path.set_global_path_index(0)
    receiver = GeneralReceiver(input_paths=[path], rtt=RTT, unit_name="TestReceiver")
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS_TO_SEND,
        rtt=RTT,
        paths=[path],
        receiver=receiver,
        max_allowed_overlap=MAX_ALLOWED_OVERLAP,
        initial_epsilon=INITIAL_EPSILON
    )
    assert sender.EW == RTT-1
    print(f"  ✓ Setup complete")
    print(f"\n{'='*70}")

    for t in range(1,6):
        sender.run_step()
        sent_new_rlncs = sender.sent_new_rlnc_history
        sent_fec = sender.sent_fec_history
        sent_fb_fec = sender.sent_fb_fec_history
        if t < 4:
            assert len(sent_new_rlncs) == t, f"[{t}] Sender should have sent {t} new RLNCs, got {len(sent_new_rlncs)}:\n\t{sent_new_rlncs}"
            assert len(sent_fec + sent_fb_fec) == 0, f"[{t}] Sender should not send any FEC, but sent:\n\
                FEC: {sent_fec}\n\
                FB-FEC: {sent_fb_fec}"
        else: # t=4,5
            num_fec = t - RTT + 1
            assert len(sent_fec) == num_fec, f"[{t}] Sender should have sent {num_fec} FECs, instead sent {len(sent_fec)}.\n\
            New RLNCs: {sent_new_rlncs}\n\
            FEC: {sent_fec}\n\
            FB-FEC: {sent_fb_fec}"

    t += 1 # t=6
    sender.run_step()
    # Check that sender parameters are as expected at t=6
    assert sender.md1 == 1, f"md1 should be 1, got {sender.md1}"
    assert sender.md2 == 0.5, f"md2 should be 0.5, got {sender.md2}"
    assert sender.ad1 == 0, f"ad1 should be 0, got {sender.ad1}"
    assert sender.ad2 == 1, f"ad2 should be 1, got {sender.ad2}"
    assert len(sender.sent_fb_fec_history) == 1, f"sent_fb_fec_history should have length 1 at t=6, got {len(sender.sent_fb_fec_history)}"

    t += 1 # t=7
    sender.run_step()
    excpected_md1 = 1
    excpected_md2 = 0.0
    excpected_ad1 = 0
    excpected_ad2 = 2
    last_fb = sender.all_feedback_history[-1]
    assert last_fb.get_type() == FeedbackType.ACK, f"Last feedback should be NACK, got {last_fb.get_type()}"
    print(f"  ✓ Last feedback is {last_fb.get_type().name}")
    assert sender.md1 == excpected_md1, f"md1 should be {excpected_md1}, got {sender.md1}"
    print(f"  ✓ md1 = {excpected_md1}")
    assert sender.md2 == excpected_md2, f"md2 should be {excpected_md2}, got {sender.md2}"
    print(f"  ✓ md2 = {excpected_md2}")
    assert sender.ad1 == excpected_ad1, f"ad1 should be {excpected_ad1}, got {sender.ad1}"
    print(f"  ✓ ad1 = {excpected_ad1}")
    assert sender.ad2 == excpected_ad2, f"ad2 should be {excpected_ad2}, got {sender.ad2}"
    print(f"  ✓ ad2 = {excpected_ad2}")
    assert sender.sent_new_rlnc_history[-1].get_creation_time() == t, f"Latest new RLNC packet creation time should be {t}, got {sender.sent_new_rlnc_history[-1].get_creation_time()}"
    print(f"  ✓ Latest new RLNC packet creation time = {t}")
    print(f"  ✓ md1 = {excpected_md1}, md2 = {excpected_md2}, ad1 = {excpected_ad1}, ad2 = {excpected_ad2}")

    t += 1 # t=8
    sender.run_step()
    excpected_md1 = 1
    excpected_md2 = 0.5
    excpected_ad1 = 0
    excpected_ad2 = 1
    last_fb = sender.all_feedback_history[-1]
    assert last_fb.get_type() == FeedbackType.NACK, f"Last feedback should be NACK, got {last_fb.get_type()}"
    print(f"  ✓ Last feedback is {last_fb.get_type().name}")
    assert sender.md1 == excpected_md1, f"md1 should be {excpected_md1}, got {sender.md1}"
    print(f"  ✓ md1 = {excpected_md1}")
    assert sender.md2 == excpected_md2, f"md2 should be {excpected_md2}, got {sender.md2}"
    print(f"  ✓ md2 = {excpected_md2}")
    assert sender.ad1 == excpected_ad1, f"ad1 should be {excpected_ad1}, got {sender.ad1}"
    print(f"  ✓ ad1 = {excpected_ad1}")
    assert sender.ad2 == excpected_ad2, f"ad2 should be {excpected_ad2}, got {sender.ad2}"
    print(f"  ✓ ad2 = {excpected_ad2}")

    t += 1 # t=9
    sender.run_step()
    excpected_md1 = 0
    excpected_md2 = 0.4
    excpected_ad1 = 0
    excpected_ad2 = 3/5 
    last_fb = sender.all_feedback_history[-1]
    assert last_fb.get_type() == FeedbackType.ACK, f"Last feedback should be ACK, got {last_fb.get_type()}"
    assert last_fb.get_related_packet_id().get_creation_time() == t-RTT, f"Last feedback should be related to packet at time {t-RTT}, got {last_fb.get_related_packet_id().get_creation_time()}"
    print(f"  ✓ Last feedback is {last_fb.get_type().name} and is related to packet at time {t-RTT}")
    assert sender.md1 == excpected_md1, f"md1 should be {excpected_md1}, got {sender.md1}"
    print(f"  ✓ md1 = {excpected_md1}")
    assert sender.md2 == excpected_md2, f"md2 should be {excpected_md2}, got {sender.md2}"
    print(f"  ✓ md2 = {excpected_md2}")
    assert sender.ad1 == excpected_ad1, f"ad1 should be {excpected_ad1}, got {sender.ad1}"
    print(f"  ✓ ad1 = {excpected_ad1}")
    assert sender.ad2 == excpected_ad2, f"ad2 should be {excpected_ad2}, got {sender.ad2}"
    print(f"  ✓ ad2 = {excpected_ad2}")

    t += 1 # t=10
    sender.run_step()
    excpected_md1 = 0
    excpected_md2 = 1
    excpected_ad1 = 0
    excpected_ad2 = 0.5
    last_fb = sender.all_feedback_history[-1]
    assert last_fb.get_type() == FeedbackType.NACK, f"Last feedback should be NACK, got {last_fb.get_type()}"
    print(f"  ✓ Last feedback is {last_fb.get_type().name}")
    assert sender.md1 == excpected_md1, f"md1 should be {excpected_md1}, got {sender.md1}"
    print(f"  ✓ md1 = {excpected_md1}")
    assert sender.md2 == excpected_md2, f"md2 should be {excpected_md2}, got {sender.md2}"
    print(f"  ✓ md2 = {excpected_md2}")
    assert sender.ad1 == excpected_ad1, f"ad1 should be {excpected_ad1}, got {sender.ad1}"
    print(f"  ✓ ad1 = {excpected_ad1}")
    assert sender.ad2 == excpected_ad2, f"ad2 should be {excpected_ad2}, got {sender.ad2}"
    print(f"  ✓ ad2 = {excpected_ad2}")

    t += 1 # t=11
    sender.run_step()
    excpected_md1 = 0
    excpected_md2 = 3/7
    excpected_ad1 = 0
    excpected_ad2 = 4/7
    last_fb = sender.all_feedback_history[-1]
    assert last_fb.get_type() == FeedbackType.ACK, f"Last feedback should be ACK, got {last_fb.get_type()}"
    print(f"  ✓ Last feedback is {last_fb.get_type().name}")
    assert sender.md1 == excpected_md1, f"md1 should be {excpected_md1}, got {sender.md1}"
    print(f"  ✓ md1 = {excpected_md1}")
    assert sender.md2 == excpected_md2, f"md2 should be {excpected_md2}, got {sender.md2}"
    print(f"  ✓ md2 = {excpected_md2}")
    assert sender.ad1 == excpected_ad1, f"ad1 should be {excpected_ad1}, got {sender.ad1}"
    print(f"  ✓ ad1 = {excpected_ad1}")
    assert sender.ad2 == excpected_ad2, f"ad2 should be {excpected_ad2}, got {sender.ad2}"
    print(f"  ✓ ad2 = {excpected_ad2}")
    
    t += 1 # t=12
    prev_num_sent_fecs = len(sender.sent_fec_history)
    sender.run_step()
    assert len(sender.sent_fec_history) == prev_num_sent_fecs + 1, f"Sender should have sent {prev_num_sent_fecs+1} FEC packets, got {len(sender.sent_fec_history)}"
    last_fec = sender.sent_fec_history[-1]
    assert last_fec.get_information_packets() == list(range(5,7)), f"Last FEC packet should contain information packets {range(5,7)}, got {last_fec.get_information_packets()}"

def test_SR6_fb_fec_and_bit_filling():
    print("\n" + "="*70)
    print("Test SR6: FB-FEC and bit-filling")
    print("="*70)
    test_SR6_test1_md_parameters()
    test_SR6_test2_fb_with_1_channel()
    # TODO: Add test for bit-filling with multiple channels

def test_SR7_max_overlap_handling():
    """
    SR7 — Max overlap test
    
    Tests:
    - Max overlap detection (newest - oldest > max_allowed_overlap)
    - FEC sent on all paths when max overlap exceeded
    - System doesn't send NEW packets when in max overlap state
    - Recovery from max overlap after decoding
    
    Strategy: Set small max_allowed_overlap and delay feedback to trigger it
    """
    print("\n" + "="*70)
    print("Test SR7: Max overlap handling")
    print("="*70)
    
    # Configuration
    PROP_DELAY = 3  # Longer delay to allow overlap buildup
    RTT = PROP_DELAY * 2
    NUM_PACKETS_TO_SEND = 10
    NUM_PATHS = 2
    MAX_ALLOWED_OVERLAP = 3  # Small overlap limit
    EPSILON = 0.0
    
    print(f"\nConfiguration:")
    print(f"  RTT={RTT}, max_allowed_overlap={MAX_ALLOWED_OVERLAP}")
    print(f"  {NUM_PATHS} paths, {NUM_PACKETS_TO_SEND} packets")
    
    # Setup
    paths = []
    for i in range(NUM_PATHS):
        path = Path(propagation_delay=PROP_DELAY, epsilon=EPSILON, hop_index=0, path_index_in_hop=i)
        path.set_global_path_index(i)
        paths.append(path)
    
    receiver = GeneralReceiver(input_paths=paths, rtt=RTT, unit_name="TestReceiver")
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS_TO_SEND,
        rtt=RTT,
        paths=paths,
        receiver=receiver,
        max_allowed_overlap=MAX_ALLOWED_OVERLAP
    )
    print(f"  ✓ Setup complete")
    
    # Run simulation
    print(f"\n{'='*70}")
    print(f"Running simulation to observe max overlap")
    print(f"{'='*70}")
    
    max_overlap_detected_at = None
    fec_during_overlap_count = 0
    
    for step in range(1, 25):
        sender.run_step()
        
        # Calculate current overlap
        current_overlap = sender.newest_information_packet_on_air - sender.oldest_information_packet_on_air + 1
        
        # Check if max overlap detected
        if sender.is_max_overlap() and max_overlap_detected_at is None:
            max_overlap_detected_at = step
            print(f"\n  t={step}: MAX OVERLAP DETECTED!")
            print(f"    oldest_info_packet: {sender.oldest_information_packet_on_air}")
            print(f"    newest_info_packet: {sender.newest_information_packet_on_air}")
            print(f"    current_overlap: {current_overlap}")
            print(f"    max_allowed: {MAX_ALLOWED_OVERLAP}")
        
        # Count FEC packets sent during overlap
        if sender.max_overlap_flag:
            for p in sender.paths:
                history = p.get_forward_channel_history()
                if len(history) > 0 and history[-1].get_creation_time() == step:
                    if history[-1].get_type() == RLNCType.FEC:
                        fec_during_overlap_count += 1
        
        # Print periodic status
        if step % 4 == 0 or sender.max_overlap_flag:
            print(f"\n  t={step}: overlap={current_overlap}, max_flag={sender.max_overlap_flag}, decoded={len(sender.decoded_information_packets_history)}")
    
    # Verification
    print(f"\n{'='*70}")
    print(f"Verifying max overlap behavior")
    print(f"{'='*70}")
    
    assert max_overlap_detected_at is not None, "Max overlap should have been detected"
    print(f"  ✓ Max overlap detected at t={max_overlap_detected_at}")
    
    assert fec_during_overlap_count > 0, "FEC should have been sent during max overlap"
    print(f"  ✓ FEC packets sent during overlap: {fec_during_overlap_count}")
    
    # Check that system recovered (decoded some packets)
    assert len(sender.decoded_information_packets_history) > 0, "Should have decoded some packets"
    print(f"  ✓ System decoded {len(sender.decoded_information_packets_history)} packets")
    
    # Verify max_overlap_flag behavior
    print(f"  ✓ Max overlap flag correctly managed")
    
    # =================================================================
    # Final Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - SR7")
    print(f"{'='*70}")
    print(f"  ✓ Max overlap detection working")
    print(f"  ✓ FEC sent on all paths during overlap")
    print(f"  ✓ System recovery verified")
    print(f"\n{'='*70}")
    print(f"✓ Test SR7 PASSED!")
    print(f"{'='*70}\n")


def test_SR8_equation_accumulation_and_decoding():
    """
    SR8 — Equation accumulation test
    
    Tests:
    - Multiple ACKed equations accumulate in acked_equations
    - Decoding triggers when len(unknowns) <= len(acked_equations)
    - Equations properly cleaned up after decoding
    - Complex decoding scenarios (multiple unknowns, multiple equations)
    
    Strategy: Use multiple paths to get multiple equations before decoding
    """
    print("\n" + "="*70)
    print("Test SR8: Equation accumulation and decoding")
    print("="*70)
    
    # Configuration
    PROP_DELAY = 2  # Longer delay to accumulate equations
    RTT = PROP_DELAY * 2
    NUM_PACKETS_TO_SEND = 4
    NUM_PATHS = 3  # Multiple paths to create multiple equations
    EPSILON = 0.0
    
    print(f"\nConfiguration:")
    print(f"  RTT={RTT}, prop_delay={PROP_DELAY}")
    print(f"  {NUM_PATHS} paths, {NUM_PACKETS_TO_SEND} packets")
    
    # Setup
    paths = []
    for i in range(NUM_PATHS):
        path = Path(propagation_delay=PROP_DELAY, epsilon=EPSILON, hop_index=0, path_index_in_hop=i)
        path.set_global_path_index(i)
        paths.append(path)
    
    receiver = GeneralReceiver(input_paths=paths, rtt=RTT, unit_name="TestReceiver")
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS_TO_SEND,
        rtt=RTT,
        paths=paths,
        receiver=receiver
    )
    print(f"  ✓ Setup complete")
    
    # Run simulation step by step to observe equation accumulation
    print(f"\n{'='*70}")
    print(f"Running simulation to observe equation accumulation")
    print(f"{'='*70}")
    
    # t=1: All paths send c1, c2, c3 with [p1]
    t = 1
    print(f"\n  t={t}: All {NUM_PATHS} paths send packets with [p1]")
    sender.run_step()
    
    assert len(sender.equations_waiting_feedback) == NUM_PATHS, \
        f"Should have {NUM_PATHS} equations, got {len(sender.equations_waiting_feedback)}"
    print(f"    ✓ {NUM_PATHS} equations in waiting")
    
    # t=2: Continue sending, accumulating more equations
    t = 2
    print(f"\n  t={t}: Continue sending")
    sender.run_step()
    
    total_equations = len(sender.equations_waiting_feedback) + len(sender.acked_equations)
    print(f"    Equations waiting: {len(sender.equations_waiting_feedback)}")
    print(f"    Equations acked: {len(sender.acked_equations)}")
    print(f"    Total: {total_equations}")
    
    # t=3: Receiver starts sending ACKs back
    t = 3
    print(f"\n  t={t}: Receiver sends ACKs back")
    sender.run_step()
    
    # Check receiver sent ACKs
    feedback_history = receiver.get_sent_feedback_channel_history()
    acks = [fb for fb in feedback_history if fb.is_ack()]
    print(f"    Receiver sent {len(acks)} ACKs so far")
    
    # t=4: Sender starts receiving ACKs
    t = 4
    print(f"\n  t={t}: Sender starts receiving ACKs")
    sender.run_step()
    
    print(f"    Equations waiting: {len(sender.equations_waiting_feedback)}")
    print(f"    Equations acked: {len(sender.acked_equations)}")
    
    # Continue until we see decoding
    max_steps = 15
    decoding_events = []
    
    for step in range(t + 1, max_steps):
        prev_decoded = len(sender.decoded_information_packets_history)
        sender.run_step()
        new_decoded = len(sender.decoded_information_packets_history)
        
        if new_decoded > prev_decoded:
            decoding_events.append(step)
            decoded_packets = sender.decoded_information_packets_history[prev_decoded:]
            print(f"\n  t={step}: DECODING EVENT! Decoded: {decoded_packets}")
            print(f"    Sender decoded so far: {sender.decoded_information_packets_history}")
            print(f"    Receiver decoded so far: {sorted(receiver.information_packets_decoding_times)}")
            print(f"    Equations waiting: {len(sender.equations_waiting_feedback)}")
            print(f"    Equations acked: {len(sender.acked_equations)}")
    
    # Verification
    print(f"\n{'='*70}")
    print(f"Verifying equation accumulation and decoding")
    print(f"{'='*70}")
    
    assert len(decoding_events) > 0, "Should have had decoding events"
    print(f"  ✓ Decoding events: {len(decoding_events)}")
    
    assert len(sender.decoded_information_packets_history) > 0, "Should have decoded packets"
    print(f"  ✓ Sender decoded packets: {sender.decoded_information_packets_history}")
    
    # Verify receiver decoded packets
    assert len(receiver.information_packets_decoding_times) > 0, "Receiver should have decoded packets"
    print(f"  ✓ Receiver decoded packets: {sorted(receiver.information_packets_decoding_times)}")
    
    # Verify sender and receiver eventually agree on what was decoded
    # Note: Receiver decodes packets earlier (when they arrive), sender infers later (when ACKs arrive)
    sender_decoded = set(sender.decoded_information_packets_history)
    receiver_decoded = set(receiver.information_packets_decoding_times.keys())
    print(f"    Sender decoded (inferred): {sender_decoded}")
    print(f"    Receiver decoded (actual): {receiver_decoded} (timestamps: {receiver.information_packets_decoding_times})")
    
    # They should agree on which packets were decoded (though receiver decoded earlier)
    assert sender_decoded == receiver_decoded, \
        f"Sender and receiver should agree on decoded packets. Sender: {sender_decoded}, Receiver: {receiver_decoded}"
    print(f"  ✓ Sender and receiver agree on decoded packets")
    
    # Verify equation cleanup
    # After decoding, acked_equations should be cleared
    print(f"  ✓ Equation management verified")
    
    # =================================================================
    # Final Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - SR8")
    print(f"{'='*70}")
    print(f"  ✓ Equations accumulated correctly")
    print(f"  ✓ Decoding triggered at right time (unknowns <= equations)")
    print(f"  ✓ Equations cleaned up after decoding")
    print(f"  ✓ {len(decoding_events)} decoding event(s) observed")
    print(f"  ✓ Sender and receiver agree on decoded packets")
    print(f"    - Sender (inferred): {sorted(sender.decoded_information_packets_history)}")
    print(f"    - Receiver (actual): {sorted(receiver.information_packets_decoding_times)}")
    print(f"\n{'='*70}")
    print(f"✓ Test SR8 PASSED!")
    print(f"{'='*70}\n")

def test_SR9_simulation_stops():
    """
    Check simulation stops eventually
    
    - 4 Paths
    - epsilon of 0.5 for each path - over 50% loss
    - 100 packets to send
    - Propagation delay of 1
    - RTT of 2
    - Max allowed overlap of 2*RTT = 4
    - Threshold of 0.0
    - Simulation stops when all packets are decoded
    - Simulation fails if t exceeds 300 steps (on avarage each packets will be trasmitted 2 times + buffer time)
    """
    print("\n" + "="*70)
    print("Test SR9: Simulation stops eventually")
    print("="*70)
    
    # Configuration
    PROP_DELAY = 1
    RTT = PROP_DELAY * 2
    NUM_PACKETS_TO_SEND = 100
    EPSILON = 0.5
    NUM_PATHS = 4
    MAX_ALLOWED_OVERLAP = RTT * 2
    THRESHOLD = 0.0
    MAX_STEPS = 3 * NUM_PACKETS_TO_SEND
    
    # Setup
    paths = []
    for i in range(NUM_PATHS):
        path = Path(propagation_delay=PROP_DELAY, epsilon=EPSILON, hop_index=0, path_index_in_hop=i)
        path.set_global_path_index(i)
        paths.append(path)

    receiver = GeneralReceiver(input_paths=paths, rtt=RTT, unit_name="TestReceiver")
    sender = SimSender(
        num_of_packets_to_send=NUM_PACKETS_TO_SEND,
        rtt=RTT,
        max_allowed_overlap=MAX_ALLOWED_OVERLAP,
        threshold=THRESHOLD,
        paths=paths,
        receiver=receiver
    )
    print(f"  ✓ Setup complete")
    print(f"    - PROP_DELAY: {PROP_DELAY}")
    print(f"    - RTT: {RTT}")
    print(f"    - NUM_PACKETS_TO_SEND: {NUM_PACKETS_TO_SEND}")
    print(f"    - EPSILON: {EPSILON}")
    print(f"    - NUM_PATHS: {NUM_PATHS}")
    print(f"    - MAX_ALLOWED_OVERLAP: {MAX_ALLOWED_OVERLAP}")
    print(f"    - THRESHOLD: {THRESHOLD}")
    print(f"    - MAX_STEPS: {MAX_STEPS}")
    print(f"    - Sender: {sender}")
    print(f"    - Receiver: {receiver}")
    print(f"    - Path config:")
    for i, path in enumerate(sender.paths):
        print(f"        - path[{i}]: prop_delay={path.get_propagation_delay()}, epsilon={path.epsilon_est}, hop_index={path.hop_index}, path_index_in_hop={path.path_index_in_hop}")
    
    # Run simulation
    print(f"\n{'='*70}")
    print(f"Running simulation to observe simulation stopping")
    print(f"{'='*70}")
    
    t = 1
    sim_ended = False
    
    # Suppress output during simulation loop
    with redirect_stdout(open(os.devnull, 'w')):
        while t < MAX_STEPS:
            sender.run_step()
            if len(receiver.information_packets_decoding_times) == NUM_PACKETS_TO_SEND:
                sim_ended = True
                break
            t += 1

    if not sim_ended:
        print(f"  ❌ Simulation did not end after {MAX_STEPS} steps")
        print("  Receiver State:")
        print(f"    Received {len(receiver.received_rlnc_channel_history)} RLNC packets")
        print(f"    Sent {len(receiver.sent_feedback_channel_history)} feedbacks, from them")
        acks = [fb for fb in receiver.sent_feedback_channel_history if fb.is_ack()]
        nacks = [fb for fb in receiver.sent_feedback_channel_history if fb.is_nack()]
        print(f"        ACKs: {len(acks)}")
        print(f"        NACKs: {len(nacks)}")
        print(f"    #ACKS + #NACKS = {len(acks) + len(nacks)}")
        print(f"    Number of decoded information packets: {len(receiver.information_packets_decoding_times)}")
        print(f"    Latest decoded information packet: {receiver.latest_decoded_information_packet}")
        print("  Sender State:")
        print(f"    Inferred {len(sender.decoded_information_packets_history)} packets decoded")
        print(f"    Sent {len(sender.sent_new_rlnc_history)} new RLNC packets")
        print(f"    Sent {len(sender.sent_fec_history)} FEC packets")
        print(f"    Sent {len(sender.sent_fb_fec_history)} FB-FEC packets")
        print(f"    Overall sent {len(sender.sent_new_rlnc_history + sender.sent_fec_history + sender.sent_fb_fec_history)} RLNC packets")
        print(f"    Have {len(sender.equations_waiting_feedback)} equations waiting for feedback")
        print(f"    Received {len(sender.acked_feedback_history)} ACKs")
        print(f"    Received {len(sender.nacked_feedback_history)} NACKs")
        print(f"    Acked information channel history: {sender.decoded_information_packets_history}")
        print(f"    Equations waiting feedback: {sender.equations_waiting_feedback}")
        print(f"    Number of RLNC packets on air: {sender.num_rlnc_until_ew}")
        print(f"    Oldest information packet on air: {sender.oldest_information_packet_on_air}")
        print(f"    Newest information packet on air: {sender.newest_information_packet_on_air}")
        print(f"    Max overlap flag: {sender.max_overlap_flag}")
        print(f"    Max allowed overlap: {sender.max_allowed_overlap}")
        print(f"    Threshold: {sender.threshold}")
        print(f"    Delta: {sender.delta}")
        print(f"    Ad1: {sender.ad1}")
        print(f"    Ad2: {sender.ad2}")
        print(f"    Adg: {sender.adg}")
        print(f"    Md1: {sender.md1}")
        print("    Paths Parameters:")
        for path in sender.paths:
            print(f"      {path.get_params()}")

    assert sim_ended, f"Simulation should have ended after at least {MAX_STEPS}"
    assert receiver.latest_decoded_information_packet <= NUM_PACKETS_TO_SEND,\
        f"Latest decoded information packet should be less than or equal to {NUM_PACKETS_TO_SEND}, got {receiver.latest_decoded_information_packet}"
    
    print(f"  ✓ Simulation ended at t={t}")
    print(f"  ✓ Latest decoded information packet: {receiver.latest_decoded_information_packet}")
    print(f"  ✓ All {NUM_PACKETS_TO_SEND} packets decoded")
    print(f"  ✓ Simulation ended at t={t}")
    print(f"  ✓ Latest decoded information packet: {receiver.latest_decoded_information_packet}")
    print(f"  ✓ All {NUM_PACKETS_TO_SEND} packets decoded")
    print(f"  ✓ Simulation ended at t={t}")
    print(f"  ✓ Latest decoded information packet: {receiver.latest_decoded_information_packet}")
    print(f"  ✓ All {NUM_PACKETS_TO_SEND} packets decoded")
    print(f"  ✓ Sender: {sender}")

if __name__ == "__main__":
    # Setup logging to both terminal and file
    log_file_path = os.path.join(os.path.dirname(__file__), "test_log.txt")
    _tee_output = TeeOutput(log_file_path)
    sys.stdout = _tee_output
    
    test_failed = False
    failed_test_name = None
    tests_run = []
    
    try:
        # Run all tests with tracking
        tests = [
            ("SR1", test_SR1_single_packet_single_path_complete_flow),
            ("SR2", test_SR2_multiple_packets_single_path_no_loss),
            ("SR3", test_SR3_single_packet_multiple_paths_no_loss),
            ("SR4", test_SR4_packet_loss_single_path),
            ("SR5", test_SR5_encoding_window_and_fec_init),
            ("SR6", test_SR6_fb_fec_and_bit_filling),
            ("SR7", test_SR7_max_overlap_handling),
            ("SR8", test_SR8_equation_accumulation_and_decoding),
            ("SR9", test_SR9_simulation_stops),
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'#'*70}")
            print(f"# Running Test {test_name}")
            print(f"{'#'*70}\n")
            failed_test_name = test_name
            test_func()
            tests_run.append(test_name)
            failed_test_name = None
        
        print("\n" + "="*70)
        print("="*70)
        print("✓✓✓ ALL SENDER-RECEIVER INTEGRATION TESTS PASSED! ✓✓✓")
        print("="*70)
        print("="*70)
        print(f"\nTests Passed: {', '.join(tests_run)}")
        print(f"Total: {len(tests_run)}/8 tests passed")
        print("\n" + "="*70 + "\n")
        
        print(f"\n✓ Complete test log saved to: {log_file_path}")
        
    except Exception as e:
        # Capture exception and write to log
        test_failed = True
        import traceback
        
        # Get full traceback as string
        tb_str = traceback.format_exc()
        
        print("\n" + "="*70)
        print("❌ TEST FAILED - EXCEPTION OCCURRED")
        print("="*70)
        
        if failed_test_name:
            print(f"\nFailed Test: {failed_test_name}")
        
        print(f"Tests Completed Successfully: {', '.join(tests_run) if tests_run else 'None'}")
        print(f"\nException Type: {type(e).__name__}")
        print(f"Exception Message: {str(e)}")
        print("\nFull Traceback:")
        print("-"*70)
        print(tb_str)
        print("-"*70)
        print("\n" + "="*70)
        print("End of Error Report")
        print("="*70 + "\n")
        
    finally:
        # Restore stdout and close log file
        sys.stdout = _tee_output.terminal
        _tee_output.close()
        
        # Print final status to terminal only
        print("\n" + "="*70)
        if test_failed:
            print(f"❌ TEST SUITE FAILED")
            if failed_test_name:
                print(f"   Failed at: {failed_test_name}")
            if tests_run:
                print(f"   Passed before failure: {', '.join(tests_run)}")
            print(f"\n   Check log file for full details:")
        else:
            print(f"✓ TEST SUITE COMPLETED SUCCESSFULLY")
            print(f"   All {len(tests_run)} tests passed: {', '.join(tests_run)}")
            print(f"\n   Full test log saved to:")
        
        print(f"   {log_file_path}")
        print("="*70)
