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
        prop_delay=2,
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
    kwargs = dict(
        num_packets_to_send=20,
        max_iterations=500,
        prop_delay=2,
        window=3,
    )
    direct = SRNetwork(
        path_epsilons=[0.0],
        num_paths=1,
        independent=True,
        **kwargs,
    )
    multihop_api = SRMpMhNetwork(
        path_epsilons=[[0.0]],
        num_paths=1,
        num_hops=1,
        in_order_forwarding=True,
        **kwargs,
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
