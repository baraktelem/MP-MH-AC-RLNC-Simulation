"""Regression tests for chain-aware SR-ARQ relay windows."""

import os
import random
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "mp_mh_network"))

from mp_mh_network.Channels import Path
from sr_arq.SRNetwork import SRMpMhNetwork, SRNetwork
from sr_arq.SRNode import SRNodeReceiver, SRNodeSender


def _make_relay_sender(*, gid: int = 2, num_chains: int = 4, window: int = 2):
    input_path = Path(2, 0.0, 0, gid - 1)
    output_path = Path(2, 0.0, 1, gid - 1)
    input_path.set_global_path_index(gid)
    output_path.set_global_path_index(gid)
    receiver = SRNodeReceiver(
        input_paths=[input_path],
        rtt=4,
        num_chains=num_chains,
    )
    sender = SRNodeSender(
        output_path=output_path,
        node_receiver=receiver,
        rtt=4,
        num_chains=num_chains,
        window=window,
    )
    return receiver, sender


def test_relay_window_uses_chain_stride_and_opens_after_ack():
    receiver, sender = _make_relay_sender(gid=2, num_chains=4, window=2)
    receiver.received_seqs.update({2, 6, 10})

    assert sender.send_base == 2
    assert sender._next_seq_to_send() == 2
    assert sender._next_seq_to_send() == 6
    assert sender._next_seq_to_send() is None

    sender.acked_seqs.add(2)
    sender._advance_send_base()

    assert sender.send_base == 6
    assert sender._next_seq_to_send() == 10


def test_relay_retransmission_is_allowed_when_window_is_full():
    receiver, sender = _make_relay_sender(gid=1, num_chains=1, window=2)
    receiver.received_seqs.update({1, 2, 3})

    assert sender._next_seq_to_send() == 1
    assert sender._next_seq_to_send() == 2
    assert sender._next_seq_to_send() is None

    sender.retransmit_queue.add(1)
    assert sender._next_seq_to_send() == 1
    assert sender._next_seq_to_send() is None


def test_multihop_relays_never_exceed_their_window():
    random.seed(7)
    window = 3
    net = SRMpMhNetwork(
        path_epsilons=[[0.0, 0.7, 0.2], [0.2, 0.4, 0.6]],
        num_paths=2,
        num_hops=3,
        global_prop_delay=6,  # end-to-end; H=3 -> hop_prop_delay=2, hop_rtt=4
        num_packets_to_send=40,
        max_iterations=2_000,
        window=window,
        in_order_forwarding=True,
    )

    for t in range(1, 2_001):
        net.t = t
        net._tick()
        for chain_nodes in net.nodes:
            for node in chain_nodes:
                outstanding = node.my_sender.forwarded - node.my_sender.acked_seqs
                assert len(outstanding) <= window
        if len(net.receiver.information_packets_decoding_times) == 40:
            break

    assert len(net.receiver.information_packets_decoding_times) == 40


def test_h1_behavior_matches_direct_single_hop_network():
    common = dict(
        num_packets_to_send=20,
        max_iterations=500,
        window=3,
    )
    # Single-hop SRNetwork keeps prop_delay (global == hop at H=1); the multi-hop
    # API takes global_prop_delay. At H=1 both equal 2 -> identical timing.
    direct = SRNetwork(
        path_epsilons=[0.0],
        num_paths=1,
        independent=True,
        prop_delay=2,
        **common,
    )
    multihop_api = SRMpMhNetwork(
        path_epsilons=[[0.0]],
        num_paths=1,
        num_hops=1,
        in_order_forwarding=True,
        global_prop_delay=2,
        **common,
    )

    direct.run_sim()
    multihop_api.run_sim()

    assert (
        direct.receiver.information_packets_decoding_times
        == multihop_api.receiver.information_packets_decoding_times
    )
    assert (
        direct.sender.inforamtion_packets_first_transmission_times
        == multihop_api.sender.inforamtion_packets_first_transmission_times
    )


def test_fixed_horizon_snapshots_throughput_then_drains_delay_cohort():
    net = SRMpMhNetwork(
        path_epsilons=[[0.0]],
        num_paths=1,
        num_hops=1,
        global_prop_delay=1,
        num_packets_to_send=float("inf"),
        max_iterations=5,
        window=None,
        in_order_forwarding=False,
    )

    net.run_sim()

    # Slots 1..5 admit five packets.  With one slot of propagation, only four
    # are delivered at the measurement horizon; the fifth arrives while the
    # frozen cohort drains at slot 6.
    assert net.measurement_horizon == 5
    assert net.delivered_at_horizon == 4
    assert net.normalized_throughput == 4 / 5
    assert net.admitted_at_horizon == frozenset({1, 2, 3, 4, 5})
    assert net.receiver_num_information_packets_decoded == 5
    assert net.t == 6
    assert net.drain_slots == 1
    assert net.inorder_delay_mean == 1
    assert net.inorder_delay_max == 1


def test_fixed_horizon_uses_one_common_throughput_denominator():
    net = SRMpMhNetwork(
        path_epsilons=[[0.0], [0.0]],
        num_paths=2,
        num_hops=1,
        global_prop_delay=1,
        num_packets_to_send=100,
        max_iterations=10,
        window=None,
    )

    # Different last-delivery times would produce a different result under
    # sum(delivered_c / finish_c). Horizon mode must use total / common T.
    net.measurement_horizon = 10
    net.delivered_at_horizon = 7
    net.receiver.chain_delivered_count = {1: 5, 2: 2}
    net.receiver.chain_finish_time = {1: 6, 2: 9}

    net.calculate_normalized_throughput_stats()

    assert net.normalized_throughput == 7 / 10


def test_equal_packets_per_path_quota_admits_exactly_k_per_chain():
    """Each chain admits exactly K new packets when packets_per_path=K."""
    random.seed(11)
    K = 15
    P = 2
    net = SRMpMhNetwork(
        path_epsilons=[[0.1, 0.3], [0.5, 0.2]],
        num_paths=P,
        num_hops=2,
        global_prop_delay=4,  # end-to-end; H=2 -> hop_prop_delay=2, hop_rtt=4
        packets_per_path=K,
        max_iterations=None,  # run until all quota packets are delivered
        window=4,
        in_order_forwarding=False,
    )

    assert net.num_packets_to_send == P * K
    assert net.sender.packets_per_path == K

    net.run_sim()

    for i in range(P):
        assert net.sender.path_admitted_count[i] == K
    assert sum(net.sender.path_admitted_count.values()) == P * K
    assert len(net.receiver.information_packets_decoding_times) == P * K
    for gid in range(1, P + 1):
        assert net.receiver.chain_delivered_count.get(gid, 0) == K

    expected_throughput = sum(
        K / net.receiver.chain_finish_time[gid] for gid in range(1, P + 1)
    )
    assert net.normalized_throughput == expected_throughput
    assert net.measurement_horizon is None
