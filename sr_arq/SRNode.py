import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))

from mp_mh_network.Receiver import GeneralReceiver, ReceiverPath
from mp_mh_network.Packet import RLNCPacket
from mp_mh_network.Channels import Path

from sr_arq.SRSender import SRSender
from sr_arq.sr_feedback import SRFeedbackMode


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

    Flow control (`node_queue_size`): the number of packets this node may still
    be holding -- received but not yet ACKed by the next hop, INCLUDING packets
    parked in the in-order reorder buffer. When that held backlog is full, a new
    arrival is refused (a NACK is sent instead of an ACK) so the upstream keeps
    the packet and retransmits it later -- i.e. hop-by-hop backpressure that stops
    the source from outrunning a downstream bottleneck. A slot frees only when the
    next hop ACKs the packet (true store-and-forward buffer). `node_queue_size=None`
    is an unbounded buffer (no backpressure); the node then behaves like a plain
    GeneralReceiver.

    This node handles a single chain; its seqs stride by `num_chains` and its
    first seq equals the input path's global_path_index (the chain id).
    """

    def __init__(
        self,
        input_paths: list[Path],
        rtt: int,
        num_chains: int = 1,
        in_order_forwarding: bool = False,
        node_queue_size: int = None,
        unit_name: str = None,
        feedback_mode: SRFeedbackMode = SRFeedbackMode.HBH,
        debug: bool = False,
    ):
        if unit_name is None:
            unit_name = "SRNodeReceiver"
        super().__init__(input_paths, rtt, unit_name, debug=debug)
        # E2E_FORWARD_ONLY turns the relay into a best-effort forwarder: this
        # receiver just captures the arrival with no upstream ACK/NACK. HBH and
        # E2E_FULL_ARQ both run the full per-hop SR-ARQ below.
        self.feedback_mode = feedback_mode
        self.in_order_forwarding = in_order_forwarding
        # Seqs made forwardable to the node's sender (out-of-order: all arrivals;
        # in-order: only the contiguously-released prefix).
        self.received_seqs: set[int] = set()

        # In-order release state (this node's chain strides by num_chains and
        # starts at the input path's global_path_index).
        self._stride = num_chains
        self._expected = input_paths[0].get_global_path_index()
        self._reorder_buffer: set[int] = set()

        # Flow control: bounded receive buffer (None = unbounded, no backpressure).
        self.node_queue_size = node_queue_size
        # Reference to the node sender's `acked_seqs` set (seqs the NEXT hop has
        # ACKed), wired by SRNode so the held (received-but-not-yet-ACKed) backlog
        # can be measured. Only used when node_queue_size is set.
        self._peer_acked: set[int] | None = None

    def run_step(self, time: int = None):
        # E2E_FORWARD_ONLY: best-effort relay input -- capture this tick's single
        # arrival with NO upstream feedback and NO received_seqs bookkeeping (the
        # node forwards the raw arrival immediately; see SRNode.run_step).
        if self.feedback_mode == SRFeedbackMode.E2E_FORWARD_ONLY:
            if time is not None:
                self.t = time
            else:
                self.t += 1
            self.arrived_packet = None
            for receiver_path in self.receiver_paths:
                arrived_packets = receiver_path.pop_arrived_packets()
                assert arrived_packets is None or len(arrived_packets) <= 1, (
                    f"Only 1 packet can be in receiver path buffer, got {arrived_packets}"
                )
                if arrived_packets:
                    self.arrived_packet = arrived_packets[0]
                    self.received_rlnc_channel_history.append(self.arrived_packet)
                    receiver_path.update_receiving_packets_strating_time(self.arrived_packet, self.t)
            return
        # With an unbounded buffer, behave exactly like the base receiver.
        if self.node_queue_size is None:
            return super().run_step(time)
        # Bounded buffer: refuse a new arrival (NACK instead of ACK) whenever the
        # held backlog (reorder-buffered + received-but-not-yet-next-hop-ACKed) is
        # full, so the upstream retransmits it once room frees up. This is the
        # hop-by-hop backpressure.
        if time is not None:
            self.t = time
        else:
            self.t += 1
        self.arrived_packet = None
        for receiver_path in self.receiver_paths:
            arrived_packets = receiver_path.pop_arrived_packets()
            assert arrived_packets is None or len(arrived_packets) <= 1, (
                f"Only 1 packet can be in receiver path buffer, got {arrived_packets}"
            )
            if not arrived_packets:
                self.send_nack(receiver_path)
                continue
            pkt = arrived_packets[0]
            seq = pkt.get_information_packets()[0]
            acked = self._peer_acked if self._peer_acked is not None else set()
            # Total packets still held: reorder-buffered (in-order mode) plus
            # released-but-not-yet-next-hop-ACKed. A slot frees only on next-hop ACK.
            backlog = len(self._reorder_buffer) + len(self.received_seqs) - len(acked)
            # Always accept duplicates/already-released seqs (so the upstream stops
            # retransmitting), AND the head-of-line seq == _expected even when full:
            # once the reorder buffer counts toward the cap, refusing the very seq
            # the node is waiting for would deadlock it. Only genuinely new seqs
            # past the in-order frontier are throttled.
            duplicate = (seq in self.received_seqs) or (self.in_order_forwarding and seq <= self._expected)
            if backlog >= self.node_queue_size and not duplicate:
                self.send_nack(receiver_path)  # backpressure: looks like a loss upstream
                continue
            self.arrived_packet = pkt
            self.received_rlnc_channel_history.append(pkt)
            receiver_path.update_receiving_packets_strating_time(pkt, self.t)
            self.send_ack(receiver_path, pkt)
            self._after_rlnc_arrived(receiver_path, pkt)

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
        num_chains: int = 1,
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
        # This relay owns one round-robin slice of the global sequence space:
        # chain gid owns gid, gid+P, gid+2P, ... .  The inherited source sender
        # uses stride 1, so replace its base with this chain's first sequence.
        self.send_stride = num_chains
        self.send_base = output_path.get_global_path_index()

    def _advance_send_base(self):
        """Advance the relay window along this chain's strided sequence space."""
        while self.send_base in self.acked_seqs:
            self.send_base += self.send_stride

    def _next_seq_to_send(self) -> int | None:
        # Downstream-NACKed seqs first (lowest), then the lowest received-but-not-
        # yet-forwarded seq inside this chain's sliding window.  Retransmissions
        # remain eligible when the window is full because they do not add a new
        # outstanding sequence.
        if self.retransmit_queue:
            seq = min(self.retransmit_queue)
            self.retransmit_queue.discard(seq)
            return seq
        pending = self.node_receiver.received_seqs - self.forwarded
        if self.window is not None:
            window_end = self.send_base + self.window * self.send_stride
            pending = {seq for seq in pending if seq < window_end}
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
        node_queue_size: int = None,
        feedback_mode: SRFeedbackMode = SRFeedbackMode.HBH,
        debug: bool = False,
    ):
        self.hop_num = hop_num
        self.unit_name = unit_name if unit_name is not None else f"SRNode[{hop_num}]"
        self.rtt = rtt
        self.input_path = input_path
        self.output_path = output_path
        self.feedback_mode = feedback_mode

        self.my_receiver = SRNodeReceiver(
            input_paths=[input_path],
            rtt=rtt,
            num_chains=num_chains,
            in_order_forwarding=in_order_forwarding,
            node_queue_size=node_queue_size,
            unit_name=f"{self.unit_name}.Receiver",
            feedback_mode=feedback_mode,
            debug=debug,
        )
        # The sender stays a full SR-ARQ output link (HBH default). In
        # E2E_FORWARD_ONLY it is used only as a plain transmitter (run_step is not
        # called; SRNode.run_step forwards via _send_seq_on_path directly).
        self.my_sender = SRNodeSender(
            output_path=output_path,
            node_receiver=self.my_receiver,
            rtt=rtt,
            num_chains=num_chains,
            unit_name=f"{self.unit_name}.Sender",
            window=window,
            debug=debug,
        )
        # Let the receiver measure its held backlog (received but not yet ACKed by
        # the next hop) for flow control -- a slot frees only on the next-hop ACK.
        self.my_receiver._peer_acked = self.my_sender.acked_seqs

    def run_step(self, time: int = None):
        # E2E_FORWARD_ONLY: best-effort store-and-forward. Take this tick's arrival
        # (no upstream feedback) and forward it once on the output path, re-stamping
        # the chain's global id and a fresh creation_time. Duplicates ARE re-forwarded
        # so that source retransmissions propagate. No per-hop retransmit queue/window.
        if self.feedback_mode == SRFeedbackMode.E2E_FORWARD_ONLY:
            self.my_receiver.run_step(time)
            self.my_sender.t = self.my_receiver.t
            out_path = self.my_sender.paths[0]
            if self.my_receiver.arrived_packet is not None:
                seq = self.my_receiver.arrived_packet.get_information_packets()[0]
                self.my_sender._send_seq_on_path(out_path, seq)
            else:
                # Nothing to forward: still advance the output forward channel so
                # in-flight packets keep propagating.
                out_path.run_forward_channel_step(current_time=self.my_sender.t)
            return
        # HBH and E2E_FULL_ARQ: full per-hop SR-ARQ (receive+ACK/NACK, then forward).
        self.my_receiver.run_step(time)
        self.my_sender.run_step(time)

    def __repr__(self) -> str:
        return (
            f"{self.unit_name}: received={len(self.my_receiver.received_seqs)}, "
            f"forwarded={len(self.my_sender.forwarded)}"
        )
