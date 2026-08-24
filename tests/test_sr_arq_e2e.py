"""Tests for the two end-to-end (E2E) feedback variants of multi-hop SR-ARQ.

Covers:
- E2E_FORWARD_ONLY (Type A, paper Fig. 19 top): forward-only relays, slot-based
  end-to-end feedback -> round-trip timing, one feedback per chain per tick,
  lossless + light-loss full decode.
- E2E_FULL_ARQ (Type B, mp_mh-style): full per-hop SR-ARQ relays, seq-based
  selective-repeat end-to-end feedback -> light-loss full decode, no spurious
  retransmission when lossless, and retransmission paced to once per e2e RTT.
- H=1: both E2E variants reduce to HBH (identical delivery / first-transmission).
"""

import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "mp_mh_network"))

from sr_arq.SRNetwork import SRMpMhNetwork
from sr_arq.sr_feedback import SRFeedbackMode


def _net(matrix, *, num_paths, num_hops, gpd, mode, packets_per_path=None,
         window=None, max_iterations=None, in_order_forwarding=False):
    return SRMpMhNetwork(
        path_epsilons=matrix,
        num_paths=num_paths,
        num_hops=num_hops,
        global_prop_delay=gpd,
        packets_per_path=packets_per_path,
        max_iterations=max_iterations,
        window=window,
        in_order_forwarding=in_order_forwarding,
        feedback_mode=mode,
    )


# ---------------------------------------------------------------------------
# Type A - E2E_FORWARD_ONLY
# ---------------------------------------------------------------------------

def test_type_a_e2e_roundtrip_timing():
    """A seq first sent at t0 is ACKed at the source exactly t0 + global_rtt later
    (a full end-to-end round trip); forward-only relays give a deterministic delay."""
    net = _net([[0.0, 0.0, 0.0]], num_paths=1, num_hops=3, gpd=6,
               mode=SRFeedbackMode.E2E_FORWARD_ONLY, packets_per_path=1, window=None)
    assert net.global_rtt == 12
    ack_tick = None
    for t in range(1, 60):
        net.t = t
        net._tick()
        if 1 in net.sender.acked_seqs:
            ack_tick = t
            break
    send_tick = net.sender.inforamtion_packets_first_transmission_times[1]
    assert send_tick == 1, f"seq 1 should first be sent at t=1, got {send_tick}"
    assert ack_tick is not None, "source never received the end-to-end ACK"
    assert ack_tick - send_tick == net.global_rtt, (
        f"E2E round trip {ack_tick - send_tick} should equal global_rtt {net.global_rtt}"
    )


def test_type_a_one_feedback_per_chain_ack_and_nack():
    """After warm-up the receiver emits exactly one slot-based feedback per chain:
    an ACK for a chain that delivered this tick, a NACK for every other chain."""
    net = _net([[0.0, 0.0, 0.0]] * 4, num_paths=4, num_hops=3, gpd=6,
               mode=SRFeedbackMode.E2E_FORWARD_ONLY, packets_per_path=1)
    rx = net.receiver
    rx.t = 100  # past the warm-up (t > e2e_prop_delay)
    rx._e2e_arrivals = [(1, 1)]  # only chain 1 delivered a packet this tick
    rx._send_e2e_feedbacks_slot()
    for gid, ch in rx.e2e_feedback_channels.items():
        assert len(ch.packets_in_channel) == 1, (
            f"chain {gid} should get exactly one feedback, got {len(ch.packets_in_channel)}"
        )
    assert rx.e2e_feedback_channels[1].packets_in_channel[0].is_ack()
    for gid in (2, 3, 4):
        assert rx.e2e_feedback_channels[gid].packets_in_channel[0].is_nack()


def test_type_a_full_decode_lossless():
    """Forward-only + lossless: every packet is delivered end-to-end and, since
    nothing is erased, the source transmits each seq exactly once (no retransmit)."""
    P, K = 4, 12
    net = _net([[0.0, 0.0, 0.0]] * P, num_paths=P, num_hops=3, gpd=6,
               mode=SRFeedbackMode.E2E_FORWARD_ONLY, packets_per_path=K,
               window=None, max_iterations=20000)
    net.run_sim()
    assert len(net.receiver.information_packets_decoding_times) == P * K
    assert len(net.sender.sent_new_rlnc_history) == P * K


def test_type_a_full_decode_light_loss():
    """Forward-only + light per-hop loss: the source retransmits end-to-end (a
    packet is delivered only if it survives all H hops) and every packet arrives."""
    random.seed(7)
    P, K = 4, 15
    matrix = [[0.1, 0.2, 0.1], [0.2, 0.1, 0.2], [0.1, 0.3, 0.1], [0.2, 0.1, 0.3]]
    net = _net(matrix, num_paths=P, num_hops=3, gpd=6,
               mode=SRFeedbackMode.E2E_FORWARD_ONLY, packets_per_path=K,
               window=22, max_iterations=50000)
    net.run_sim()
    assert len(net.receiver.information_packets_decoding_times) == P * K


# ---------------------------------------------------------------------------
# Type B - E2E_FULL_ARQ
# ---------------------------------------------------------------------------

def test_type_b_full_decode_light_loss():
    """Full per-hop SR-ARQ relays + seq-based end-to-end feedback: every packet is
    delivered (first-hop losses recovered end-to-end, other hops recovered per-hop)."""
    random.seed(11)
    P, K = 4, 15
    matrix = [[0.2, 0.2, 0.1], [0.3, 0.1, 0.2], [0.1, 0.3, 0.2], [0.2, 0.2, 0.3]]
    net = _net(matrix, num_paths=P, num_hops=3, gpd=6,
               mode=SRFeedbackMode.E2E_FULL_ARQ, packets_per_path=K,
               window=22, max_iterations=50000, in_order_forwarding=True)
    net.run_sim()
    assert len(net.receiver.information_packets_decoding_times) == P * K


def test_type_b_no_retransmission_when_lossless():
    """With no erasures anywhere, the seq-based feedback / RTT-paced retransmission
    must not fire: the source transmits each seq exactly once (ACKs arrive before
    the retransmission timeout, so no in-flight packet is re-sent)."""
    P, K = 3, 10
    net = _net([[0.0, 0.0, 0.0]] * P, num_paths=P, num_hops=3, gpd=6,
               mode=SRFeedbackMode.E2E_FULL_ARQ, packets_per_path=K,
               window=22, max_iterations=20000, in_order_forwarding=True)
    net.run_sim()
    assert len(net.receiver.information_packets_decoding_times) == P * K
    assert len(net.sender.sent_new_rlnc_history) == P * K, (
        "lossless E2E_FULL_ARQ must not retransmit any packet"
    )


def test_type_b_retransmission_paced_by_e2e_rtt():
    """First-hop losses force end-to-end retransmission; each seq must be
    (re)transmitted at most once per end-to-end RTT (no re-sending in-flight
    packets), and all packets are still delivered."""
    random.seed(3)
    P, K = 2, 15
    matrix = [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]  # only the first hop is lossy
    net = _net(matrix, num_paths=P, num_hops=3, gpd=6,
               mode=SRFeedbackMode.E2E_FULL_ARQ, packets_per_path=K,
               window=22, max_iterations=50000, in_order_forwarding=True)
    net.run_sim()
    assert len(net.receiver.information_packets_decoding_times) == P * K

    # There must be genuine retransmissions (more transmissions than packets).
    assert len(net.sender.sent_new_rlnc_history) > P * K

    # Consecutive (re)transmissions of the same seq must be >= global_rtt apart.
    send_times: dict[int, list[int]] = {}
    for pkt in net.sender.sent_new_rlnc_history:
        send_times.setdefault(pkt.get_seq(), []).append(pkt.get_creation_time())
    retransmitted = 0
    for seq, times in send_times.items():
        times.sort()
        for a, b in zip(times, times[1:]):
            retransmitted += 1
            assert b - a >= net.global_rtt, (
                f"seq {seq} re-sent after {b - a} < global_rtt {net.global_rtt}"
            )
    assert retransmitted > 0, "expected some retransmissions on a lossy first hop"


# ---------------------------------------------------------------------------
# H=1: both E2E variants reduce to HBH
# ---------------------------------------------------------------------------

def test_h1_e2e_equals_hbh_lossless():
    """With a single hop there are no relays, so all three modes are the same
    single-hop SR-ARQ and produce identical delivery / first-transmission maps."""
    P, K = 3, 12
    results = {}
    for mode in (SRFeedbackMode.HBH, SRFeedbackMode.E2E_FORWARD_ONLY, SRFeedbackMode.E2E_FULL_ARQ):
        net = _net([[0.0]] * P, num_paths=P, num_hops=1, gpd=3,
                   mode=mode, packets_per_path=K, window=None, max_iterations=20000)
        net.run_sim()
        results[mode] = (
            dict(net.sender.inforamtion_packets_first_transmission_times),
            dict(net.receiver.information_packets_decoding_times),
        )
    base_first, base_dec = results[SRFeedbackMode.HBH]
    for mode in (SRFeedbackMode.E2E_FORWARD_ONLY, SRFeedbackMode.E2E_FULL_ARQ):
        first, dec = results[mode]
        assert first == base_first, f"{mode.name}: first-transmission map differs from HBH"
        assert dec == base_dec, f"{mode.name}: delivery map differs from HBH"


if __name__ == "__main__":
    test_type_a_e2e_roundtrip_timing()
    test_type_a_one_feedback_per_chain_ack_and_nack()
    test_type_a_full_decode_lossless()
    test_type_a_full_decode_light_loss()
    test_type_b_full_decode_light_loss()
    test_type_b_no_retransmission_when_lossless()
    test_type_b_retransmission_paced_by_e2e_rtt()
    test_h1_e2e_equals_hbh_lossless()
    print("All SR-ARQ E2E tests passed.")
