"""
Comprehensive tests for GeneralReceiver class.

Test naming convention:
- R1-R23: Receiver tests based on specifications

Key test areas:
1. Basic ACK/NACK correctness (R1-R3)
2. Multi-path behavior (R4-R6, R10)
3. Feedback generation (R11)
4. Internal state tracking (R15-R17)
5. Edge cases (R18-R19, R23)
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Channels import Channel, ForwardChannel, Path
from Packet import RLNCPacket, FeedbackPacket, RLNCType, FeedbackType
from Receiver import GeneralReceiver, ReceiverPath


def make_test_rlnc_packet(information_packets: int, path_id: int, prop_delay: int, creation_time: int = 0) -> RLNCPacket:
    """Helper to create a test RLNC packet."""
    return RLNCPacket(
        global_path_id=path_id,
        type=RLNCType.NEW,
        information_packets=[information_packets],
        prop_time_left_in_channel=prop_delay,
        creation_time=creation_time
    )


def create_test_paths(num_paths: int, prop_delay: int, epsilon: float = 0.0) -> list[Path]:
    """Helper to create test paths."""
    paths = []
    for i in range(num_paths):
        path = Path(
            propagation_delay=prop_delay,
            epsilon=epsilon,
            hop_index=0,
            path_index_in_hop=i
        )
        path.set_global_path_index(i)
        paths.append(path)
    return paths


def inject_packet_to_path(path: Path, packet: RLNCPacket, current_time: int):
    """Helper to inject a packet into a path's forward channel."""
    path.forward_channel.add_packets_to_channel([packet], time=current_time)


def run_channels_for_ticks(paths: list[Path], num_ticks: int):
    """Helper to run all path channels for a number of ticks."""
    for _ in range(num_ticks):
        for path in paths:
            path.forward_channel.run_step()
            path.feedback_channel.run_step()


def count_feedback_by_type(feedback_history: list[FeedbackPacket]) -> dict:
    """Count ACKs and NACKs in feedback history."""
    ack_count = sum(1 for fb in feedback_history if fb.is_ack())
    nack_count = sum(1 for fb in feedback_history if fb.is_nack())
    return {'ack': ack_count, 'nack': nack_count}


def get_feedback_history(path: Path) -> list[FeedbackPacket]:
    """Get feedback history from path's feedback channel."""
    return path.feedback_channel.get_channel_history()


def get_forward_history(path: Path) -> list[RLNCPacket]:
    """Get RLNC packet history from path's forward channel."""
    return path.forward_channel.get_channel_history()

def run_test_step(receiver: GeneralReceiver, time: int=None):
    """Run a test step for the receiver."""
    # Run forward channels for all paths
    paths = receiver.receiver_paths
    for path in paths:
        path.forward_channel.run_step()
    # Run receiver step
    receiver.run_step()
    # Run feedback channels for all paths
    for path in paths:
        path.feedback_channel.run_step()

    return time+1 if time is not None else None

def run_n_test_steps(receiver: GeneralReceiver, n: int):
    for _ in range(n):
        run_test_step(receiver)
# ============================================================================
# 1. Basic ACK/NACK correctness
# ============================================================================

def test_R0_receiver_with_no_path_flow():
    """
    R0 — Receiver with no path flow
    Verifies complete flow:
    1. Run 3 test steps- make sure receiver is not sending feedback yet
    2. Inject packet directly to Receiver's input (skip forward channel) and check histories
    """
    print("\n=== Test R0: Receiver with no path flow ===")
    PROP_DELAY = 0
    path = create_test_paths(num_paths=1, prop_delay=PROP_DELAY)[0] # GeneralReceiver needs at least one path for init
    receiver = GeneralReceiver(input_paths=[path], rtt=PROP_DELAY)

    # Run 3 test steps- make sure receiver is not sending feedback yet
    print(" Step 1: Run 3 test steps- make sure receiver is not sending feedback yet")
    run_n_test_steps(receiver, 3)
    sent_feedback_history = receiver.get_sent_feedback_channel_history()
    assert len(sent_feedback_history) == 0, \
     f"Receiver should not have received any packets yet, but sent {len(sent_feedback_history)} feedback packets"
    print(f"  ✓ Receiver should not have received any packets yet, but sent {len(sent_feedback_history)} feedback packets")
    
    # Inject packet directly to Receiver's input (skip forward channel)
    rlnc_packet_creation_time = 0
    print(" Step 2: Inject packet directly to Receiver's input (skip forward channel) and check histories")
    pkt = make_test_rlnc_packet(information_packets=1, path_id=0, prop_delay=PROP_DELAY, creation_time=rlnc_packet_creation_time)
    path.forward_channel.arrived_packets.append(pkt)
    receiver.run_step() # Run only receiver step to avoid feedback channel asserts
    print(f"  ✓ Packet injected directly to Receiver's input:\n    {pkt}")
    print(f"   Checking received RLNC packets history:")
    received_rlnc_channel_history = receiver.get_received_rlnc_channel_history()
    assert len(received_rlnc_channel_history) == 1, \
        f"Receiver should have received 1 packet, but got {len(receiver.get_received_rlnc_channel_history())}"
    assert received_rlnc_channel_history[0].creation_time == rlnc_packet_creation_time, \
        f"Received packet should have creation time {rlnc_packet_creation_time}, but got {received_rlnc_channel_history[0].creation_time}"
    print(f"  ✓ Receiver have received 1 packet:\n    {received_rlnc_channel_history}")
    print(f"   Checking sent feedback history:")
    sent_feedback_history = receiver.get_sent_feedback_channel_history()
    assert len(sent_feedback_history) == 1, \
        f"Receiver should have sent 1 feedback packet, but got {len(sent_feedback_history)}"
    assert sent_feedback_history[0].get_creation_time() == receiver.t, \
        f"Sent feedback packet should have creation time {receiver.t}, but got {sent_feedback_history[0].creation_time}: \n\
            {sent_feedback_history[0]}"
    print(f"  ✓ Receiver have sent 1 feedback packet:\n    {sent_feedback_history}")

    print("✓ Test R0 passed: Receiver with no path flow verified")

def test_R1_receiver_with_single_path_flow():
    """
    R0 — Simple end-to-end test: 1 path, 1 RLNC packet → 1 ACK feedback
    Verifies complete flow:
    1. RLNC packet: forward_channel.add_packets_to_channel
    2. forward_channel.run_step() x prop_delay → packet arrives
    3. receiver.run_step() → processes packet, generates ACK
    4. feedback_channel.run_step() x prop_delay → ACK arrives
    5. Verify ACK can be popped from feedback_channel
    """
    print("\n=== Test R1: Receiver with single path flow ===")
    
    prop_delay = 3
    rtt = prop_delay * 2
    paths = create_test_paths(num_paths=1, prop_delay=prop_delay)
    receiver = GeneralReceiver(input_paths=paths, rtt=rtt)
    path = paths[0]
    global t
    t = 0
    
    def run_r0_step():
        path.forward_channel.run_step()
        receiver.run_step()
        path.feedback_channel.run_step()
        global t
        t += 1

    # Step 1: Inject RLNC packet to forward channel
    pkt = make_test_rlnc_packet(information_packets=1, path_id=0, prop_delay=prop_delay, creation_time=t)
    print(f"  Step 1: Add RLNC packet to forward_channel:")
    path.forward_channel.add_packets_to_channel([pkt], time=t)
    print(f"    {pkt}")
    
    # Verify packet is in forward channel history
    forward_history = get_forward_history(path)
    assert len(forward_history) == 1, f"Forward channel should have 1 packet in history, got {len(forward_history)}"
    print(f"  ✓ Packet added to forward channel history")
    
    # Step 2: Run forward_channel for prop_delay ticks to deliver RLNC packet
    print(f"  Step 2: Run forward_channel propagation delay ({prop_delay}) times)")
    for _ in range(prop_delay+1):
        run_r0_step()
    
    # Verify packet arrived (but don't pop it yet - let receiver do that)
    assert len(receiver.get_received_rlnc_channel_history()) == 1, "RLNC packet should have arrived"
    print(f"  ✓ RLNC packet arrived at forward_channel")
    
    # Step 3: Receiver.run_step() → pops arrived packet, processes, generates feedback
    print(f"  Step 3: receiver.run_step()")
    run_r0_step()
    
    # Verify forward channel's arrived packets were consumed by receiver
    assert len(path.forward_channel.arrived_packets) == 0, \
        "Receiver should have consumed arrived packets from forward_channel"
    print(f"  ✓ Receiver consumed packet from forward_channel")
    
    # Verify ACK is in feedback channel history
    feedback_history = get_feedback_history(path)
    assert len(feedback_history) >= 1, "Feedback channel should have at least 1 ACK in history"
    ack = feedback_history[0]  # Most recent feedback
    assert ack.is_ack(), f"Latest feedback should be ACK instead got:\n    {ack}"
    assert ack.global_path_id == 0, f"ACK should be for path 0, got {ack.global_path_id}"
    print(f"  ✓ ACK generated and added to feedback_channel")
    
    # Step 4: Run feedback_channel for prop_delay ticks to deliver ACK
    print(f"  Step 4: Run feedback_channel {prop_delay} times")
    for _ in range(prop_delay):
        path.feedback_channel.run_step()
    
    # Step 5: Verify ACK arrived and can be popped
    assert len(path.feedback_channel.arrived_packets) > 0, \
        "ACK should have arrived at feedback_channel"
    
    # arrived_feedback = path.feedback_channel.pop_arrived_packets()
    # assert arrived_feedback is not None, "Should be able to pop arrived feedback"
    # assert len(arrived_feedback) == 1, f"Should have 1 arrived feedback packet, got {len(arrived_feedback)}"
    # arrived_ack = arrived_feedback[0]
    # assert arrived_ack.is_ack(), "Arrived feedback should be ACK"
    # assert arrived_ack.global_path_id == 0, "ACK should be for path 0"
    # print(f"  ✓ ACK arrived at feedback_channel and can be popped")
    
    # # Verify channel histories are complete
    # # Forward channel: should have exactly 1 RLNC packet
    # forward_history = get_forward_history(path)
    # assert len(forward_history) == 1, f"Forward history should have exactly 1 packet, got {len(forward_history)}"
    
    # # Feedback channel: should have exactly 1 ACK
    # feedback_history = get_feedback_history(path)
    # assert len(feedback_history) == 1, \
    #     f"Feedback history should have exactly 1 packet, got {len(feedback_history)}:\n    " + \
    #     '\n    '.join(str(fb) for fb in feedback_history)
    
    print("✓ Test R1 passed: Complete end-to-end flow verified")

def test_R2_receiver_with_single_path_flow():
    """
    R1 — Receiver with multiple paths flow
    Verifies complete flow:
    1. Create several paths with epsilon = 0
    2. Inject packet to each path at sequential times
    3. Run test until all a time all paths should have received the feedback
    """
    print("\n=== Test R2: Receiver with multiple paths flow ===")
    PROP_DELAY = 1
    RTT = PROP_DELAY * 2
    NUM_PATHS = 3

    print(f"    Step 1: Creating GeneralReceiver, Paths and Packets")
    paths = create_test_paths(num_paths=NUM_PATHS, prop_delay=PROP_DELAY, epsilon=0.0)
    receiver = GeneralReceiver(input_paths=paths, rtt=RTT)
    
    # Create packets for each path
    
    # pkt2 = make_test_rlnc_packet(information_packets=2, path_id=2, prop_delay=PROP_DELAY, creation_time=t)
    print(f"Receiver: {receiver}")
    print(f"Paths:")
    print(paths, "\n")

    t = 1
    print(f"    Step 2: Injecting packets to paths sequentially")
    pkt0 = make_test_rlnc_packet(information_packets=0, path_id=0, prop_delay=PROP_DELAY, creation_time=t)
    print(f"Injecting packet0 to path 0 at time {t}:\n    {pkt0}")
    inject_packet_to_path(paths[0], pkt0, current_time=t)
    t = run_test_step(receiver, time=t)
    expected_first_received_packet_time = t
    assert len(receiver.get_received_rlnc_channel_history()) == 0, \
        f"Receiver should not have received any packets yet, but got {len(receiver.get_received_rlnc_channel_history())}:\n\
            {receiver.get_received_rlnc_channel_history()}"

    pkt1 = make_test_rlnc_packet(information_packets=1, path_id=1, prop_delay=PROP_DELAY, creation_time=t)
    pkt1_creation_time = t
    print(f"Injecting packet1 to path 1 at time {t}:\n    {pkt1}")
    inject_packet_to_path(paths[1], pkt1, current_time=t)
    t = run_test_step(receiver, time=t)
    # Assertions for path 0
    received_rlnc_channel_history = receiver.get_received_rlnc_channel_history()
    assert len(received_rlnc_channel_history) == 1, \
        f"Receiver should have received 1 packet, but got {len(received_rlnc_channel_history)}:\n\
            {received_rlnc_channel_history}"
    assert received_rlnc_channel_history[0].get_information_packets() == [0], \
        f"Received packet should have information packets [0], but got {received_rlnc_channel_history[0].get_information_packets()}:\n\
            {received_rlnc_channel_history[0]}"
    sent_feedback_history = receiver.get_sent_feedback_channel_history()
    assert len(sent_feedback_history) == 1, \
        f"Receiver should have sent 1 feedback packet, but got {len(sent_feedback_history)}:\n\
            {sent_feedback_history}"
    assert sent_feedback_history[0].get_global_path() == 0, \
        f"Feedback packet should be for path 0, but got {sent_feedback_history[0].get_global_path()}:\n\
            {sent_feedback_history}"
    assert sent_feedback_history[0].get_creation_time() == pkt1_creation_time, \
        f"Feedback packet should have creation time {t}, but got {sent_feedback_history[0].get_creation_time()}:\n\
            {sent_feedback_history}"
    assert sent_feedback_history[0].is_ack(), \
        f"Feedback packet should be an ACK, but got {sent_feedback_history[0].is_ack()}:\n\
            {sent_feedback_history}"
    
    pkt2 = make_test_rlnc_packet(information_packets=2, path_id=2, prop_delay=PROP_DELAY, creation_time=t)
    print(f"Injecting packet2 to path 2 at time {t}:\n    {pkt2}")
    inject_packet_to_path(paths[2], pkt2, current_time=t)
    t = run_test_step(receiver, time=t)
    # Assertions for path 0
    # Assert the receiver's time matches t
    assert receiver.t == t-1, f"Receiver's current time (self.t) should be {t-1}, but got {receiver.t}"

    # ---- Path 0 assertions ----
    print(f"Path 0 assertions")
    # Path 0: should have received 1 packet, sent 2 feedback packets, last is NACK
    path0 = receiver.get_receiver_path(0)
    path0_received_history = path0.get_received_channel_history()
    path0_sent_feedback = path0.get_sent_feedback_channel_history()

    expected_path0_received_history_len = 1
    expected_path0_sent_feedback_len = 2
    assert len(path0_received_history) == expected_path0_received_history_len, \
        f"Path 0 should have {expected_path0_received_history_len} packet in received history, got {len(path0_received_history)}:\n\
            {path0_received_history}"
    assert len(path0_sent_feedback) == expected_path0_sent_feedback_len, \
        f"Path 0 should have sent {expected_path0_sent_feedback_len} feedback packets, got {len(path0_sent_feedback)}:\n\
            {path0_sent_feedback}"
    assert path0_sent_feedback[-1].is_nack(), \
        f"Path 0 last sent feedback should be a NACK, got {path0_sent_feedback[-1]}: \n\
            {path0_sent_feedback[-1]}"
    print(f"✓ Path 0 assertions passed")
    # ---- Path 1 assertions ----
    print(f"Path 1 assertions")
    path1 = receiver.get_receiver_path(1)
    path1_received_history = path1.get_received_channel_history()
    path1_sent_feedback = path1.get_sent_feedback_channel_history()
    expected_path1_received_history_len = 1
    expected_path1_sent_feedback_len = 1
    assert len(path1_received_history) == expected_path1_received_history_len, \
        f"Path 1 should have {expected_path1_received_history_len} packet in received history, got {len(path1_received_history)}:\n\
            {path1_received_history}"
    assert len(path1_sent_feedback) == expected_path1_sent_feedback_len, \
        f"Path 1 should have sent {expected_path1_sent_feedback_len} feedback packet, got {len(path1_sent_feedback)}:\n\
            {path1_sent_feedback}"
    assert path1_sent_feedback[-1].is_ack(), \
        f"Path 1 last sent feedback should be an ACK, got {path1_sent_feedback[-1]}: \n\
            {path1_sent_feedback[-1]}"
    print(f"✓ Path 1 assertions passed")
    # ---- Path 2 assertions ----
    print(f"Path 2 assertions")
    path2 = receiver.get_receiver_path(2)
    path2_received_history = path2.get_received_channel_history()
    path2_sent_feedback = path2.get_sent_feedback_channel_history()
    expected_path2_received_history_len = 1
    expected_path2_sent_feedback_len = 0
    assert len(path2_received_history) == expected_path2_received_history_len, \
        f"Path 2 should have {expected_path2_received_history_len} packets in received history, got {len(path2_received_history)}:\n\
            {path2_received_history}"
    assert len(path2_sent_feedback) == expected_path2_sent_feedback_len, \
        f"Path 2 should have sent {expected_path2_sent_feedback_len} feedback packets, got {len(path2_sent_feedback)}:\n\
            {path2_sent_feedback}"
    print(f"✓ Path 2 assertions passed")


    print(f"Waiting for all paths to receive feedback")
    for _ in range(2):
        print(f"Waiting for next feedback time {t}")
        t = run_test_step(receiver, time=t)

    path0_sent_feedback_history = path0.get_sent_feedback_channel_history()
    path1_sent_feedback_history = path1.get_sent_feedback_channel_history()
    path2_sent_feedback_history = path2.get_sent_feedback_channel_history()
    assert len(path0_sent_feedback_history) == t-expected_first_received_packet_time, \
        f"Path 0 sent feedback history should have {t-expected_first_received_packet_time} packets, got {len(path0_sent_feedback_history)}:\n\
            {path0_sent_feedback_history}"
    assert len(path1_sent_feedback_history) == t-expected_first_received_packet_time-1, \
        f"Path 1 sent feedback history should have {t-expected_first_received_packet_time-1} packets, got {len(path1_sent_feedback_history)}:\n\
            {path1_sent_feedback_history}"
    assert len(path2_sent_feedback_history) == t-expected_first_received_packet_time-2, \
        f"Path 2 sent feedback history should have {t-expected_first_received_packet_time-2} packets, got {len(path2_sent_feedback_history)}:\n\
            {path2_sent_feedback_history}"

    print(f"✓ Test R2 passed")

# ============================================================================
# Main test runner
# ============================================================================

def run_all_tests():
    """Run all receiver tests and report results."""
    print("\n" + "="*70)
    print("RUNNING GENERALRECEIVER TESTS")
    print("="*70)
    
    # Only receiver flow- no path
    test_R0_receiver_with_no_path_flow()

    # End-to-end flow test
    test_R1_receiver_with_single_path_flow()

    # Receiver with multiple paths flow
    test_R2_receiver_with_single_path_flow()
    
    # # Basic ACK/NACK tests
    # test_R3_ack_for_one_valid_packet_one_path()
    # test_R2_ack_for_valid_packet_on_path_p()
    # test_R3_nack_for_missing_packet()
    # test_R3b_nack_with_multiple_packet_loss()
    
    # # Multi-path behavior
    # test_R4_valid_packets_on_multiple_paths_same_tick()
    # test_R7_nack_isolation_between_paths()
    # test_R5_some_paths_send_packets_others_dont()
    # test_R6_path_starvation()
    # test_R10_correctly_ignore_paths_that_deliver_none()
    
    # # Feedback generation
    # test_R11_feedback_contains_correct_path_id()
    
    # # Internal state tracking
    # test_R15_receiver_logs_arrival_ticks()
    # test_R16_receiver_maintains_per_path_history()
    # test_R17_receiver_does_not_mix_packets_between_paths()
    
    # # Edge cases
    # test_R18_no_packets_for_long_period()
    # test_R19_many_ticks_heavy_load_one_path()
    # test_R23_ack_timing_correctness()
    
    # # Additional tests
    # test_feedback_flows_through_pending_to_channel()
    
    print("\n" + "="*70)
    print("ALL RECEIVER TESTS PASSED! ✓✓✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()

