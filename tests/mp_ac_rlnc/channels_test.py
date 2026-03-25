"""
Comprehensive tests for Channel and ForwardChannel classes.

Test naming convention:
- C1-C8: Channel tests (delay-only, no erasures)
- F1-F8: ForwardChannel tests (delay + erasure)
"""

import sys
import os
import random

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Channels import Channel, ForwardChannel
from Packet import RLNCPacket, FeedbackPacket, RLNCType, FeedbackType, CodedEquation


def make_test_feedback_packet(packet_id: int, prop_delay: int) -> FeedbackPacket:
    """Helper to create a test feedback packet for Channel tests."""
    return FeedbackPacket(
        global_path_id=packet_id,
        type=FeedbackType.ACK,
        related_packet_id=packet_id,
        prop_time_left_in_channel=prop_delay,
        creation_time=0
    )


def make_test_rlnc_packet(packet_id: int, prop_delay: int) -> RLNCPacket:
    """Helper to create a test RLNC packet for ForwardChannel tests."""
    return RLNCPacket(
        global_path_id=packet_id,
        type=RLNCType.NEW,
        information_packets=[packet_id],
        prop_time_left_in_channel=prop_delay,
        creation_time=0
    )

def run_step(channel: Channel, time: int, packets: list[RLNCPacket] = None):
    """Run one step of the channel"""
    if packets is not None:
        channel.add_packets_to_channel(packets, time)
    channel.run_step()
    return time + 1

# ============================================================================
# Tests for Channel (delay-only, no erasures)
# ============================================================================

def test_C1_single_packet_simple_delay():
    """
    Test C1 - Single packet, simple delay
    Goal: Basic correctness of delay handling.
    """
    print("\n=== Test C1: Single packet, simple delay ===")
    
    delay = 3
    ch = Channel(propagation_delay=delay, hop_index=0, path_index_in_hop=0)
    
    # t=0: send packet
    t = 0
    pkt1 = make_test_feedback_packet(1, delay)

    # t=1: tick and check
    t = run_step(ch, t, [pkt1])
    arrived = ch.pop_arrived_packets()
    print(f"t=1: ch.packets_in_channel={ch.packets_in_channel}")
    assert arrived == ([] or None), f"t=1: Expected [], got {arrived}"
    
    # t=2: tick and check
    t = run_step(ch, t)
    arrived = ch.pop_arrived_packets()
    print(f"t=2: ch.packets_in_channel={ch.packets_in_channel}")
    assert arrived == ([] or None), f"t=2: Expected [], got {arrived}"
    
    # t=3: tick and check (packet should arrive)
    t = run_step(ch, t)
    arrived = ch.pop_arrived_packets()
    assert len(arrived) == 1, f"t=3: Expected 1 packet, got {len(arrived)}"
    assert arrived[0].global_path_id == 1, f"t=3: Expected packet 1, got {arrived[0].global_path_id}"
    
    # Verify no duplicate delivery
    t = run_step(ch, t)
    arrived = ch.pop_arrived_packets()
    assert arrived == ([] or None), f"t=4: Expected no duplicate, got {arrived}"
    
    print("✓ Test C1 passed")


def test_C2_multiple_packets_same_time():
    """
    Test C2 - Multiple packets, same injection time
    Goal: Check FIFO and batched delivery after delay.
    """
    print("\n=== Test C2: Multiple packets, same injection time ===")
    
    delay = 2
    ch = Channel(propagation_delay=delay, hop_index=0, path_index_in_hop=0)
    
    # t=0: send 3 packets
    pkt1 = make_test_feedback_packet(1, delay)
    pkt2 = make_test_feedback_packet(2, delay)
    pkt3 = make_test_feedback_packet(3, delay)
    ch.add_packets_to_channel([pkt1, pkt2, pkt3], time=0)
    
    # t=0: tick
    t = 0
    print(f"t=0: ch.packets_in_channel={ch.packets_in_channel}")
    arrived = ch.pop_arrived_packets()
    assert arrived == ([] or None), f"t=0: Expected [], got {arrived}"
    
    # t=1: tick
    ch.run_step()
    print(f"t=1: ch.packets_in_channel={ch.packets_in_channel}")
    arrived = ch.pop_arrived_packets()
    assert arrived == ([] or None), f"t=1: Expected [], got {arrived}"
    
    # t=2: tick (all should arrive)
    ch.run_step()
    print(f"t=2: ch.packets_in_channel={ch.packets_in_channel}")
    arrived = ch.pop_arrived_packets()
    assert len(arrived) == 3, f"t=2: Expected 3 packets, got {len(arrived)}:\n    {arrived}"
    
    # Check order preservation
    ids = [p.global_path_id for p in arrived]
    assert ids == [1, 2, 3], f"t=2: Expected [1,2,3], got {ids}"
    
    print("✓ Test C2 passed")


def test_C3_staggered_sends():
    """
    Test C3 – Staggered sends (interleaving)
    Goal: Check per-packet delay independent of injection time.
    """
    print("\n=== Test C3: Staggered sends ===")
    
    delay = 2
    ch = Channel(propagation_delay=delay, hop_index=0, path_index_in_hop=0)
    
    # t=0: send pkt1
    pkt1 = make_test_feedback_packet(1, delay)
    ch.add_packets_to_channel([pkt1], time=0)
    arrived = ch.pop_arrived_packets()
    assert arrived == ([] or None), f"t=0: Expected [], got {arrived}"
    
    # t=1: tick, send pkt2
    ch.run_step()
    arrived = ch.pop_arrived_packets()
    assert arrived == ([] or None), f"t=1: Expected [] (pkt1 in-flight), got {arrived}"
    
    pkt2 = make_test_feedback_packet(2, delay)
    ch.add_packets_to_channel([pkt2], time=1)
    
    # t=2: tick, send pkt3 (pkt1 should arrive)
    ch.run_step()
    arrived = ch.pop_arrived_packets()
    assert len(arrived) == 1, f"t=2: Expected 1 packet, got {len(arrived)}"
    assert arrived[0].global_path_id == 1, f"t=2: Expected pkt1, got {arrived[0].global_path_id}"
    
    pkt3 = make_test_feedback_packet(3, delay)
    ch.add_packets_to_channel([pkt3], time=2)
    
    # t=3: tick (pkt2 should arrive)
    ch.run_step()
    arrived = ch.pop_arrived_packets()
    assert len(arrived) == 1, f"t=3: Expected 1 packet, got {len(arrived)}"
    assert arrived[0].global_path_id == 2, f"t=3: Expected pkt2, got {arrived[0].global_path_id}"
    
    # t=4: tick (pkt3 should arrive)
    ch.run_step()
    arrived = ch.pop_arrived_packets()
    assert len(arrived) == 1, f"t=4: Expected 1 packet, got {len(arrived)}"
    assert arrived[0].global_path_id == 3, f"t=4: Expected pkt3, got {arrived[0].global_path_id}"
    
    print("✓ Test C3 passed")


def test_C4_zero_delay():
    """
    Test C4 – Zero delay edge case
    Goal: What happens with delay_slots=0?
    
    Design choice: With delay=0, packet should be delivered in the next tick
    (since we decrement first, 0 → -1 would violate our assertion, so we use delay=1 minimum)
    Actually, let's test delay=1 as the minimum meaningful delay.
    """
    print("\n=== Test C4: Zero/minimum delay edge case ===")
    
    # Test with delay=1 (minimum meaningful delay)
    delay = 1
    ch = Channel(propagation_delay=delay, hop_index=0, path_index_in_hop=0)
    
    # t=0: send packet
    pkt1 = make_test_feedback_packet(1, delay)
    ch.add_packets_to_channel([pkt1], time=0)
    arrived = ch.pop_arrived_packets()
    assert arrived == ([] or None), f"t=0: Expected [], got {arrived}"
    
    # t=1: tick (packet should arrive)
    ch.run_step()
    arrived = ch.pop_arrived_packets()
    assert len(arrived) == 1, f"t=1: Expected 1 packet with delay=1, got {len(arrived)}"
    
    print("✓ Test C4 passed")


def test_C5_idle_channel():
    """
    Test C5 – No packets, idle channel
    Goal: Stability in idle condition.
    """
    print("\n=== Test C5: Idle channel ===")
    
    ch = Channel(propagation_delay=5, hop_index=0, path_index_in_hop=0)
    
    # Run 100 ticks with no packets
    for t in range(100):
        ch.run_step()
        arrived = ch.pop_arrived_packets()
        assert arrived == ([] or None), f"t={t}: Expected empty channel, got {arrived}"
    
    print("✓ Test C5 passed")


def test_C6_long_simulation_memory_stability():
    """
    Test C6 – Long simulation, memory stability
    Goal: Ensure channel clears internal buffers after delivery.
    """
    print("\n=== Test C6: Long simulation, memory stability ===")
    
    delay = 5
    ch = Channel(propagation_delay=delay, hop_index=0, path_index_in_hop=0)
    
    num_packets = 1000
    max_time = 200
    
    # Send packets at random times
    random.seed(42)
    send_schedule = {}
    for i in range(num_packets):
        send_time = random.randint(0, 100)
        if send_time not in send_schedule:
            send_schedule[send_time] = []
        send_schedule[send_time].append(i)
    
    delivered_packets = []
    
    # Simulate
    for t in range(max_time):
        # Send packets scheduled for this time
        if t in send_schedule:
            for pkt_id in send_schedule[t]:
                pkt = make_test_feedback_packet(pkt_id, delay)
                ch.add_packets_to_channel([pkt], time=t)
        
        # Tick
        ch.run_step()
        arrived = ch.pop_arrived_packets()
        if arrived is not None:
            delivered_packets.extend(arrived)
    
    # Verify all packets delivered exactly once
    assert len(delivered_packets) == num_packets, \
        f"Expected {num_packets} packets, got {len(delivered_packets)}"
    
    delivered_ids = [p.global_path_id for p in delivered_packets]
    assert len(set(delivered_ids)) == num_packets, \
        f"Found duplicate deliveries"
    
    # Verify channel is empty
    assert ch.is_empty(), "Channel should be empty after all deliveries"
    
    print("✓ Test C6 passed")


def test_C7_packet_identity_preserved():
    """
    Test C7 – Packet identity not modified
    Goal: Channel should not corrupt packets.
    """
    print("\n=== Test C7: Packet identity preserved ===")
    
    delay = 3
    ch = Channel(propagation_delay=delay, hop_index=0, path_index_in_hop=0)
    
    # Create feedback packet with known fields
    pkt = FeedbackPacket(
        global_path_id=42,
        type=FeedbackType.NACK,
        related_packet_id=100,
        prop_time_left_in_channel=delay,
        creation_time=100
    )
    
    original_id = pkt.global_path_id
    original_type = pkt.type
    original_ack_id = pkt.related_packet_id
    original_creation = pkt.creation_time
    
    # Send through channel
    ch.add_packets_to_channel([pkt], time=0)
    
    # Wait for delivery
    for _ in range(delay):
        ch.run_step()
    
    arrived = ch.pop_arrived_packets()
    assert len(arrived) == 1, "Should deliver exactly one packet"
    
    delivered = arrived[0]
    
    # Verify fields unchanged
    assert delivered.global_path_id == original_id, "global_path_id corrupted"
    assert delivered.type == original_type, "type corrupted"
    assert delivered.related_packet_id == original_ack_id, "related_packet_id corrupted"
    assert delivered.creation_time == original_creation, "creation_time corrupted"
    assert delivered.prop_time_left_in_channel == 0, "prop_time should be 0 on arrival"
    
    print("✓ Test C7 passed")


def test_C8_multiple_channels_parallel():
    """
    Test C8 – Multiple channels in parallel
    Goal: Multiple channels do not interfere.
    """
    print("\n=== Test C8: Multiple channels in parallel ===")
    
    ch1 = Channel(propagation_delay=1, hop_index=0, path_index_in_hop=0)
    ch2 = Channel(propagation_delay=3, hop_index=0, path_index_in_hop=1)
    
    # t=0: send different packets through each channel
    pkt1 = make_test_feedback_packet(1, 1)
    pkt2 = make_test_feedback_packet(2, 3)
    ch1.add_packets_to_channel([pkt1], time=0)
    ch2.add_packets_to_channel([pkt2], time=0)
    
    # t=1: tick both
    ch1.run_step()
    ch2.run_step()
    
    arrived1 = ch1.pop_arrived_packets()
    arrived2 = ch2.pop_arrived_packets()
    
    assert len(arrived1) == 1, f"ch1 should deliver at t=1, got {len(arrived1)}"
    assert arrived1[0].global_path_id == 1, "ch1 delivered wrong packet"
    assert arrived2 is None or len(arrived2) == 0, f"ch2 should not deliver at t=1, got {len(arrived2)}"
    
    # t=2: tick both
    ch1.run_step()
    ch2.run_step()
    
    arrived1 = ch1.pop_arrived_packets()
    arrived2 = ch2.pop_arrived_packets()
    
    assert arrived1 is None or len(arrived1) == 0, "ch1 should not deliver at t=2"
    assert arrived2 is None or len(arrived2) == 0, "ch2 should not deliver at t=2"
    
    # t=3: tick both
    ch1.run_step()
    ch2.run_step()
    
    arrived1 = ch1.pop_arrived_packets()
    arrived2 = ch2.pop_arrived_packets()
    
    assert arrived1 is None or len(arrived1) == 0, "ch1 should not deliver at t=3"
    assert arrived2 is not None and len(arrived2) == 1, f"ch2 should deliver at t=3, got {len(arrived2)}"
    assert arrived2[0].global_path_id == 2, "ch2 delivered wrong packet"
    
    print("✓ Test C8 passed")


# ============================================================================
# Tests for ForwardChannel (delay + erasure)
# ============================================================================

def test_F1_zero_loss_channel():
    """
    Test F1 - Zero-loss channel (ε=0)
    Goal: ForwardChannel must behave exactly like Channel if ε=0.
    """
    print("\n=== Test F1: Zero-loss channel ===")
    
    delay = 3
    fch = ForwardChannel(propagation_delay=delay, epsilon=0.0, hop_index=0, path_index_in_hop=0)
    
    # Repeat test C1 with ForwardChannel
    pkt1 = make_test_rlnc_packet(1, delay)
    fch.add_packets_to_channel([pkt1], time=0)
    
    # t=0,1,2: should be empty
    for t in range(3):
        arrived = fch.pop_arrived_packets()
        assert arrived == ([] or None), f"t={t}: Expected [], got {arrived}"
        fch.run_step()
    
    # t=3: should deliver
    arrived = fch.pop_arrived_packets()
    assert len(arrived) == 1, f"t=3: Expected 1 packet, got {len(arrived)}"
    
    print("✓ Test F1 passed")


def test_F2_full_loss_channel():
    """
    Test F2 - Full-loss channel (ε=1)
    Goal: All packets are always dropped.
    """
    print("\n=== Test F2: Full-loss channel ===")
    
    delay = 3
    fch = ForwardChannel(propagation_delay=delay, epsilon=1.0, hop_index=0, path_index_in_hop=0)
    
    # Send many packets
    for i in range(100):
        pkt = make_test_rlnc_packet(i, delay)
        fch.add_packets_to_channel([pkt], time=0)
    
    # Tick for enough time
    for _ in range(delay + 10):
        fch.run_step()
        arrived = fch.pop_arrived_packets()
        assert arrived == ([] or None), f"No packets should be delivered with epsilon=1.0, got {len(arrived)}"
    
    # All packets should be dropped
    assert len(fch.dropped_packets) == 100, \
        f"Expected 100 dropped packets, got {len(fch.dropped_packets)}"
    
    print("✓ Test F2 passed")


def test_F3_statistical_correctness():
    """
    Test F3 - Statistical correctness of erasure probability
    Goal: Check that empirical loss rate ≈ ε.
    """
    print("\n=== Test F3: Statistical correctness ===")
    
    delay = 1
    epsilon = 0.3
    fch = ForwardChannel(propagation_delay=delay, epsilon=epsilon, hop_index=0, path_index_in_hop=0)
    
    N = 10000
    random.seed(42)
    
    pkt_list = []
    # Send N packets
    for i in range(N):
        pkt_list.append(make_test_rlnc_packet(i, delay))
        # fch.add_packets_to_channel([pkt])
    
    # Tick until all delivered or dropped
    run_step(fch, 0, pkt_list)
    delivered_count = 0
    for _ in range(delay + 10):
        fch.run_step()
        arrived = fch.pop_arrived_packets()
        delivered_count += len(arrived) if arrived is not None else 0
    
    # Calculate empirical loss rate
    empirical_loss_rate = 1 - (delivered_count / N)
    
    print(f"  Expected loss rate: {epsilon:.3f}")
    print(f"  Empirical loss rate: {empirical_loss_rate:.3f}")
    print(f"  Delivered: {delivered_count}/{N}")
    print(f"  Dropped: {len(fch.dropped_packets)}/{N}")
    
    # Check within tolerance
    tolerance = 0.02
    assert abs(empirical_loss_rate - epsilon) < tolerance, \
        f"Empirical loss rate {empirical_loss_rate:.3f} differs from epsilon {epsilon:.3f} by more than {tolerance}"
    
    print("✓ Test F3 passed")


def test_F4_independence_between_packets():
    """
    Test F4 - Independence between packets
    Goal: Losses should not be correlated programmatically.
    """
    print("\n=== Test F4: Independence between packets ===")
    
    delay = 1
    epsilon = 0.5
    random.seed(42)
    fch = ForwardChannel(propagation_delay=delay, epsilon=epsilon, hop_index=0, path_index_in_hop=0)
    
    N = 1000
    
    # Send N packets
    for i in range(N):
        pkt = make_test_rlnc_packet(i, delay)
        fch.add_packets_to_channel([pkt], time=0)
    
    # Collect results
    delivered = set()
    for _ in range(delay + 10):
        fch.run_step()
        arrived = fch.pop_arrived_packets()
        for p in arrived if arrived is not None else []:
            delivered.add(p.global_path_id)
    
    # Create sequence of delivered/dropped (1/0)
    sequence = [1 if i in delivered else 0 for i in range(N)]
    
    # Count runs (consecutive 0s or 1s)
    runs = []
    current_run = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    
    # Sanity check: should have both long and short runs
    assert min(runs) == 1, "Should have some single-packet runs"
    assert max(runs) > 5, "Should have some longer runs (not strictly alternating)"
    
    # Check that it's not pathologically alternating
    alternations = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i-1])
    # With 1000 packets and 50% loss, shouldn't alternate more than ~600 times
    assert alternations < 600, f"Too many alternations: {alternations} (suspiciously non-random)"
    
    print(f"  Runs statistics: min={min(runs)}, max={max(runs)}, count={len(runs)}")
    print(f"  Alternations: {alternations}")
    
    print("✓ Test F4 passed")


def test_F5_determinism_with_seed():
    """
    Test F5 - Determinism with fixed RNG seed
    Goal: Reproducibility.
    """
    print("\n=== Test F5: Determinism with fixed seed ===")
    
    delay = 2
    epsilon = 0.5
    
    def run_scenario(seed):
        random.seed(seed)
        fch = ForwardChannel(propagation_delay=delay, epsilon=epsilon, hop_index=0, path_index_in_hop=0)
        
        # Send 100 packets
        for i in range(100):
            pkt = make_test_rlnc_packet(i, delay)
            fch.add_packets_to_channel([pkt], time=0)
        
        # Collect delivered IDs
        delivered = []
        for _ in range(delay + 10):
            fch.run_step()
            arrived = fch.pop_arrived_packets()
            if arrived is not None:
                delivered.extend([p.global_path_id for p in arrived])
        
        return sorted(delivered)
    
    # Run twice with same seed
    result1 = run_scenario(123)
    result2 = run_scenario(123)
    
    assert result1 == result2, "Same seed should produce identical results"
    
    # Run with different seed should give different results (with high probability)
    result3 = run_scenario(456)
    assert result1 != result3, "Different seeds should produce different results"
    
    print("✓ Test F5 passed")


def test_F6_order_preservation():
    """
    Test F6 - Order preservation among surviving packets
    Goal: Drops should not reorder packets that survive.
    """
    print("\n=== Test F6: Order preservation ===")
    
    delay = 2
    epsilon = 0.4
    random.seed(789)
    fch = ForwardChannel(propagation_delay=delay, epsilon=epsilon, hop_index=0, path_index_in_hop=0)
    
    # Send 100 packets at the same time
    for i in range(100):
        pkt = make_test_rlnc_packet(i, delay)
        fch.add_packets_to_channel([pkt], time=0)
    
    # Collect delivered packets
    delivered_ids = []
    for _ in range(delay + 10):
        fch.run_step()
        arrived = fch.pop_arrived_packets()
        if arrived is not None:
            delivered_ids.extend([p.global_path_id for p in arrived])
    
    # Check that delivered IDs are in non-decreasing order
    for i in range(1, len(delivered_ids)):
        assert delivered_ids[i] >= delivered_ids[i-1], \
            f"Order violation: {delivered_ids[i-1]} came before {delivered_ids[i]}"
    
    print(f"  Delivered {len(delivered_ids)}/100 packets in correct order")
    print("✓ Test F6 passed")


def test_F7_multiple_channels_different_epsilon():
    """
    Test F7 – Multiple channels with different ε
    Goal: Independence across channels.
    """
    print("\n=== Test F7: Multiple channels with different epsilon ===")
    
    delay = 2
    random.seed(111)
    
    ch_good = ForwardChannel(propagation_delay=delay, epsilon=0.1, hop_index=0, path_index_in_hop=0)
    ch_bad = ForwardChannel(propagation_delay=delay, epsilon=0.8, hop_index=0, path_index_in_hop=1)
    
    N = 1000
    
    # Send same packets through both channels
    for i in range(N):
        pkt_good = make_test_rlnc_packet(i, delay)
        pkt_bad = make_test_rlnc_packet(i, delay)
        ch_good.add_packets_to_channel([pkt_good], time=0)
        ch_bad.add_packets_to_channel([pkt_bad], time=0)
    
    # Tick both channels
    delivered_good = 0
    delivered_bad = 0
    for _ in range(delay + 10):
        ch_good.run_step()
        ch_bad.run_step()
        
        arrived_good = ch_good.pop_arrived_packets()
        if arrived_good is not None:
            delivered_good += len(arrived_good)
        arrived_bad = ch_bad.pop_arrived_packets()
        if arrived_bad is not None:
            delivered_bad += len(arrived_bad)
        if ch_bad.pop_arrived_packets() is not None:
            delivered_bad += len(ch_bad.pop_arrived_packets())
    
    loss_good = 1 - (delivered_good / N)
    loss_bad = 1 - (delivered_bad / N)
    
    print(f"  Good channel (ε=0.1): delivered {delivered_good}/{N}, loss rate {loss_good:.3f}")
    print(f"  Bad channel (ε=0.8): delivered {delivered_bad}/{N}, loss rate {loss_bad:.3f}")
    
    # Good channel should deliver many more
    assert delivered_good > delivered_bad * 2, \
        "Good channel should deliver significantly more than bad channel"
    
    # Check approximate loss rates
    assert abs(loss_good - 0.1) < 0.03, f"Good channel loss rate {loss_good:.3f} != 0.1"
    assert abs(loss_bad - 0.8) < 0.03, f"Bad channel loss rate {loss_bad:.3f} != 0.8"
    
    print("✓ Test F7 passed")


def test_F8_loss_under_staggered_injection():
    """
    Test F8 – Loss under staggered injection
    Goal: Erasure logic is independent of send-time pattern.
    """
    print("\n=== Test F8: Loss under staggered injection ===")
    
    delay = 2
    epsilon = 0.3
    random.seed(999)
    fch = ForwardChannel(propagation_delay=delay, epsilon=epsilon, hop_index=0, path_index_in_hop=0)
    
    # Send packets at irregular times
    send_times = [0, 0, 2, 5, 5, 7, 10, 10, 10, 15, 20]
    packet_count = 0
    delivered_count = 0
    
    for t in range(30):
        # Send packets scheduled for this time
        if t in send_times:
            for _ in range(send_times.count(t)):
                pkt = make_test_rlnc_packet(packet_count, delay)
                fch.add_packets_to_channel([pkt], time=t)
                packet_count += 1
        
        # Tick
        fch.run_step()
        arrived = fch.pop_arrived_packets()
        delivered_count += len(arrived) if arrived is not None else 0
        
        # Verify timing: packets should arrive delay steps after being sent
        for p in arrived if arrived is not None else []:
            # Each arrived packet should have prop_time == 0
            assert p.prop_time_left_in_channel == 0
    
    # Check empirical loss rate
    empirical_loss = 1 - (delivered_count / packet_count)
    
    print(f"  Sent {packet_count} packets at staggered times")
    print(f"  Delivered {delivered_count} packets")
    print(f"  Empirical loss rate: {empirical_loss:.3f} (expected ~{epsilon:.3f})")
    
    # Verify loss rate is reasonable (with smaller sample, use larger tolerance)
    assert abs(empirical_loss - epsilon) < 0.15, \
        f"Empirical loss rate {empirical_loss:.3f} differs too much from epsilon {epsilon:.3f}"
    
    print("✓ Test F8 passed")


# ============================================================================
# Main test runner
# ============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("RUNNING CHANNEL AND FORWARDCHANNEL TESTS")
    print("="*70)
    
    # Channel tests
    test_C1_single_packet_simple_delay()
    test_C2_multiple_packets_same_time()
    test_C3_staggered_sends()
    test_C4_zero_delay()
    test_C5_idle_channel()
    test_C6_long_simulation_memory_stability()
    test_C7_packet_identity_preserved()
    test_C8_multiple_channels_parallel()
    
    # ForwardChannel tests
    test_F1_zero_loss_channel()
    test_F2_full_loss_channel()
    test_F3_statistical_correctness()
    test_F4_independence_between_packets()
    test_F5_determinism_with_seed()
    test_F6_order_preservation()
    test_F7_multiple_channels_different_epsilon()
    test_F8_loss_under_staggered_injection()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓✓✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()

