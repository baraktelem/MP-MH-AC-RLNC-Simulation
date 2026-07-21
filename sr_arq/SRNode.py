import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))

from mp_mh_network.Receiver import GeneralReceiver, ReceiverPath
from mp_mh_network.Packet import RLNCPacket
from mp_mh_network.Channels import Path

from sr_arq.SRSender import SRSender


class SRNodeReceiver(GeneralReceiver):
    """Input side of an SR-ARQ relay node (one input link).

    Reuses GeneralReceiver verbatim for the slot-based ACK (on arrival) / NACK
    (on empty slot) sent back UPSTREAM on the input path's feedback channel.

    It exposes `received_seqs`, the set of seqs the node's sender may forward.
    Two forwarding disciplines are supported:

    - in_order_forwarding=False (default): a seq becomes forwardable as soon as
      it arrives, so the node forwards out-of-order (skips gaps; the sender then
      forwards the lowest available). Efficient, low delay.
    - in_order_forwarding=True: the node runs a FULL SR-ARQ endpoint -- a seq is
      only made forwardable once all lower seqs of this chain have arrived, i.e.
      the input stream is delivered in order before being forwarded. This adds
      per-hop head-of-line blocking (higher delay), matching the paper's
      "full SR-ARQ protocol at each node".

    This node handles a single chain; its seqs stride by `num_chains` and its
    first seq equals the input path's global_path_index (the chain id).
    """

    def __init__(
        self,
        input_paths: list[Path],
        rtt: int,
        num_chains: int = 1,
        in_order_forwarding: bool = False,
        unit_name: str = None,
        debug: bool = False,
    ):
        if unit_name is None:
            unit_name = "SRNodeReceiver"
        super().__init__(input_paths, rtt, unit_name, debug=debug)
        self.in_order_forwarding = in_order_forwarding
        # Seqs made forwardable to the node's sender (out-of-order: all arrivals;
        # in-order: only the contiguously-released prefix).
        self.received_seqs: set[int] = set()

        # In-order release state (this node's chain strides by num_chains and
        # starts at the input path's global_path_index).
        self._stride = num_chains
        self._expected = input_paths[0].get_global_path_index()
        self._reorder_buffer: set[int] = set()

    def _after_rlnc_arrived(self, receiver_path: ReceiverPath, arrived_packet: RLNCPacket) -> None:
        seq = arrived_packet.get_information_packets()[0]
        if not self.in_order_forwarding:
            self.received_seqs.add(seq)
            self.sim_print(f"received seq {seq} to forward (out-of-order)")
            return
        # In-order: buffer, then release every seq contiguous (stride num_chains)
        # from the expected one into the forwardable set.
        if seq < self._expected or seq in self._reorder_buffer:
            return
        self._reorder_buffer.add(seq)
        while self._expected in self._reorder_buffer:
            self._reorder_buffer.discard(self._expected)
            self.received_seqs.add(self._expected)
            self.sim_print(f"released seq {self._expected} to forward (in-order)")
            self._expected += self._stride


class SRNodeSender(SRSender):
    """Output side of an SR-ARQ relay node (one output link).

    Runs the same SR-ARQ machinery as SRSender (retransmit queue, lowest-seq-
    first, slot->seq NACK resolution driven by DOWNSTREAM feedback), but its
    'new' packets are not a 1..N counter: they are the seqs the node's receiver
    has received but not yet forwarded. Forwarding is out-of-order (it sends the
    lowest not-yet-forwarded seq, skipping gaps; missing lower seqs arrive later
    via the upstream link's own retransmission). Chain identity is preserved
    because _send_seq_on_path stamps the output path's global_path_index (= the
    chain index) onto the forwarded DataPacket.
    """

    def __init__(
        self,
        output_path: Path,
        node_receiver: SRNodeReceiver,
        rtt: int,
        unit_name: str = None,
        window: int = None,
        debug: bool = False,
    ):
        # num_of_packets_to_send is unused (we override _next_seq_to_send); the
        # node forwards whatever it receives.
        super().__init__(
            num_of_packets_to_send=0,
            rtt=rtt,
            paths=[output_path],
            window=window,
            next_hop=None,
            debug=debug,
        )
        self.unit_name = unit_name if unit_name is not None else "SRNodeSender"
        self.node_receiver = node_receiver
        self.forwarded: set[int] = set()  # seqs sent downstream at least once

    def _next_seq_to_send(self) -> int | None:
        # Downstream-NACKed seqs first (lowest), then the lowest received-but-not-
        # yet-forwarded seq (out-of-order forwarding: gaps are skipped).
        if self.retransmit_queue:
            seq = min(self.retransmit_queue)
            self.retransmit_queue.discard(seq)
            return seq
        pending = self.node_receiver.received_seqs - self.forwarded
        if pending:
            seq = min(pending)
            self.forwarded.add(seq)
            return seq
        return None

    def __repr__(self) -> str:
        s = "SRNodeSender:"
        s += f"\n  forwarded: {len(self.forwarded)}"
        s += f"\n  retransmit queue: {len(self.retransmit_queue)}"
        s += f"\n  total transmissions: {len(self.sent_new_rlnc_history)}"
        return s


class SRNode:
    """Single-input / single-output hop-by-hop SR-ARQ relay.

    Mirrors mp_mh_network.Node (a receiver + a sender), but uncoded: it ACK/NACKs
    upstream, buffers received seqs, and forwards them downstream with its own
    SR-ARQ, retransmitting on downstream NACKs. Built one-per-(chain, hop) at the
    network level; the network orchestrates ticking, so the sender's next_hop is
    None (no cascade here).
    """

    def __init__(
        self,
        hop_num: int,
        input_path: Path,
        output_path: Path,
        rtt: int,
        unit_name: str = None,
        window: int = None,
        num_chains: int = 1,
        in_order_forwarding: bool = False,
        debug: bool = False,
    ):
        self.hop_num = hop_num
        self.unit_name = unit_name if unit_name is not None else f"SRNode[{hop_num}]"
        self.rtt = rtt
        self.input_path = input_path
        self.output_path = output_path

        self.my_receiver = SRNodeReceiver(
            input_paths=[input_path],
            rtt=rtt,
            num_chains=num_chains,
            in_order_forwarding=in_order_forwarding,
            unit_name=f"{self.unit_name}.Receiver",
            debug=debug,
        )
        self.my_sender = SRNodeSender(
            output_path=output_path,
            node_receiver=self.my_receiver,
            rtt=rtt,
            unit_name=f"{self.unit_name}.Sender",
            window=window,
            debug=debug,
        )

    def run_step(self, time: int = None):
        # Receive (and ACK/NACK upstream) first, then forward downstream.
        self.my_receiver.run_step(time)
        self.my_sender.run_step(time)

    def __repr__(self) -> str:
        return (
            f"{self.unit_name}: received={len(self.my_receiver.received_seqs)}, "
            f"forwarded={len(self.my_sender.forwarded)}"
        )
