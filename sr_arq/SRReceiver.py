import sys
import os
import copy

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))

from mp_mh_network.Receiver import GeneralReceiver, ReceiverPath
from mp_mh_network.Packet import RLNCPacket, FeedbackPacket, FeedbackType, PacketID
from mp_mh_network.Channels import Path, Channel

from sr_arq.sr_feedback import SRFeedbackMode


class SRReceiver(GeneralReceiver):
    """Selective-Repeat ARQ receiver.

    Reuses GeneralReceiver verbatim for round-robin path polling and the
    built-in slot-based ACK (on arrival) / NACK (on an empty slot) feedback.
    The only added behaviour is the reorder buffer + in-order delivery, done in
    the _after_rlnc_arrived hook. Delivery times are recorded into
    information_packets_decoding_times (the attribute name the Network stats
    pipeline reads), so throughput / in-order delay are computed unchanged.
    """

    def __init__(self, input_paths: list[Path], rtt: int, unit_name: str = None, debug: bool = False):
        if unit_name is None:
            unit_name = "SRReceiver"
        super().__init__(input_paths, rtt, unit_name, debug=debug)

        # In-order delivery state (seqs are 1..N).
        self.delivered_up_to: int = 0          # highest in-order delivered seq
        self.buffer: set[int] = set()          # arrived but not yet in-order seqs

        # Time each seq became deliverable in order (consumed by Network stats).
        self.information_packets_decoding_times: dict[int, int] = {}

    def _after_rlnc_arrived(self, receiver_path: ReceiverPath, arrived_packet: RLNCPacket) -> None:
        seq = arrived_packet.get_information_packets()[0]

        # Dedup: a late retransmission of an already-delivered/buffered seq.
        if seq <= self.delivered_up_to or seq in self.buffer:
            self.sim_print(f"duplicate seq {seq} ignored (delivered_up_to={self.delivered_up_to})")
            return

        self.buffer.add(seq)

        # Release every seq that is now contiguous from delivered_up_to. They
        # all become in-order deliverable at the current time (e.g. 6 buffered,
        # then 5 arrives -> 5 and 6 both delivered at this t).
        while (self.delivered_up_to + 1) in self.buffer:
            self.delivered_up_to += 1
            self.buffer.discard(self.delivered_up_to)
            self.information_packets_decoding_times[self.delivered_up_to] = self.t
            self.sim_print(f"delivered seq {self.delivered_up_to} in order at t={self.t}")

    def __repr__(self) -> str:
        s = super().__repr__()
        s += f"\n  delivered_up_to: {self.delivered_up_to}"
        s += f"\n  buffered (out-of-order): {len(self.buffer)}"
        if hasattr(self, "t") and self.t > 0:
            s += f"\n  normalized throughput for t{self.t}: {len(self.information_packets_decoding_times) / self.t}"
        return s


class SRSimReceiver(GeneralReceiver):
    """Multi-hop / multi-chain SR-ARQ receiver with DECOUPLED per-chain in-order
    delivery.

    Each chain is an independent flow: a packet's chain is its global_path_id
    (stable along a chain, = c+1), and the source uses a global round-robin seq
    split (chain c owns seqs {c+1, c+1+P, ...}, stride P = num_chains). This
    receiver keeps a separate in-order frontier per chain, so a stalled bad chain
    never blocks a good one. It records:
      - information_packets_decoding_times[seq] = in-order delivery time (global
        seq is unique, so the sender's first-transmission time keys the delay);
      - chain_finish_time[gid] / chain_delivered_count[gid] for the decoupled
        sum-of-per-chain-rates throughput.
    """

    def __init__(
        self,
        input_paths: list[Path],
        rtt: int,
        num_chains: int,
        unit_name: str = None,
        feedback_mode: SRFeedbackMode = SRFeedbackMode.HBH,
        e2e_feedback_channels: dict[int, Channel] = None,
        e2e_prop_delay: int = None,
        debug: bool = False,
    ):
        if unit_name is None:
            unit_name = "SRSimReceiver"
        super().__init__(input_paths, rtt, unit_name, debug=debug)
        self.stride = num_chains  # global round-robin stride (= P)

        # End-to-end feedback wiring. In both E2E modes the receiver emits a
        # per-chain end-to-end feedback (to the source) on top of the base per-hop
        # feedback (to the last node). HBH emits only the per-hop feedback.
        self.feedback_mode = feedback_mode
        self.e2e_feedback_channels = e2e_feedback_channels
        self.e2e_prop_delay = e2e_prop_delay  # global (end-to-end) one-way delay
        if feedback_mode.is_e2e():
            assert e2e_feedback_channels is not None, \
                "E2E feedback channels are required for an end-to-end feedback mode"
            assert e2e_prop_delay is not None, \
                "e2e_prop_delay (global one-way delay) is required for an end-to-end feedback mode"
        # (gid, seq) physically received this tick; drives the end-to-end ACKs.
        self._e2e_arrivals: list[tuple[int, int]] = []

        # Per-chain in-order state (chain id = global_path_id = c+1).
        self.expected: dict[int, int] = {}       # gid -> next in-order seq for that chain
        self.buffer: dict[int, set[int]] = {}    # gid -> arrived but not yet in-order seqs

        # Stats.
        self.information_packets_decoding_times: dict[int, int] = {}  # global seq -> delivery time
        self.chain_finish_time: dict[int, int] = {}                   # gid -> last delivery time
        self.chain_delivered_count: dict[int, int] = {}               # gid -> #delivered

    def run_step(self, time: int = None):
        # Base step pops arrivals (-> _after_rlnc_arrived) and sends per-hop
        # feedback to the last node (needed by HBH and by E2E_FULL_ARQ relays;
        # harmless/unused for E2E_FORWARD_ONLY relays). Then emit the per-chain
        # end-to-end feedback to the source, once all arrivals this tick are known.
        if self.feedback_mode.is_e2e():
            self._e2e_arrivals = []
        super().run_step(time)
        if self.feedback_mode == SRFeedbackMode.E2E_FORWARD_ONLY:
            self._send_e2e_feedbacks_slot()
        elif self.feedback_mode == SRFeedbackMode.E2E_FULL_ARQ:
            self._send_e2e_feedbacks_seq()
        elif self.feedback_mode == SRFeedbackMode.E2E_TIMEOUT:
            self._send_e2e_feedbacks_ack_only()

    def _after_rlnc_arrived(self, receiver_path: ReceiverPath, arrived_packet: RLNCPacket) -> None:
        gid = arrived_packet.get_global_path()
        seq = arrived_packet.get_information_packets()[0]
        # Record every physical arrival (even duplicates) for the end-to-end ACK,
        # so the source learns a seq reached the destination and stops resending it.
        if self.feedback_mode.is_e2e():
            self._e2e_arrivals.append((gid, seq))
        # First seq of chain gid is gid itself (c+1); frontier strides by P.
        exp = self.expected.setdefault(gid, gid)
        buf = self.buffer.setdefault(gid, set())

        if seq < exp or seq in buf:
            self.sim_print(f"duplicate seq {seq} on chain {gid} ignored")
            return

        buf.add(seq)
        while exp in buf:
            buf.discard(exp)
            self.information_packets_decoding_times[exp] = self.t
            self.chain_finish_time[gid] = self.t
            self.chain_delivered_count[gid] = self.chain_delivered_count.get(gid, 0) + 1
            self.sim_print(f"delivered seq {exp} (chain {gid}) in order at t={self.t}")
            exp += self.stride
        self.expected[gid] = exp

    def _send_e2e_feedbacks_slot(self) -> None:
        """E2E_FORWARD_ONLY: slot-based feedback (deterministic forward delay).

        Emit exactly one end-to-end feedback per chain per tick: ACK for a chain
        that delivered a packet this tick (carrying the seq), NACK for a chain with
        no arrival. A NACK's creation_time = t - e2e_prop_delay identifies the seq
        the source transmitted at that slot (resolved at the source)."""
        if self.t <= self.e2e_prop_delay:  # warm-up: nothing could have arrived yet
            return
        creation_time = self.t - self.e2e_prop_delay
        arrived_gids = set()
        for gid, seq in self._e2e_arrivals:
            arrived_gids.add(gid)
            self._send_e2e_feedback(gid, FeedbackType.ACK, PacketID(gid, creation_time), [seq])
        for gid in self.e2e_feedback_channels:
            if gid not in arrived_gids:
                self._send_e2e_feedback(gid, FeedbackType.NACK, PacketID(gid, creation_time), None)

    def _send_e2e_feedbacks_seq(self) -> None:
        """E2E_FULL_ARQ: seq-based selective-repeat feedback (variable delay).

        ACK every physical arrival this tick (carrying the seq) so the source stops
        resending it; NACK the genuine per-chain gaps (a chain-stride seq below the
        highest received on that chain that has not arrived). The source rate-limits
        retransmission to once per end-to-end RTT, so re-reporting the same gap each
        tick does not cause repeated resends."""
        for gid, seq in self._e2e_arrivals:
            self._send_e2e_feedback(gid, FeedbackType.ACK, PacketID(gid, self.t), [seq])
        for gid in self.e2e_feedback_channels:
            buf = self.buffer.get(gid, set())
            if not buf:
                continue
            exp = self.expected.get(gid, gid)
            hi = max(buf)
            for seq in range(exp, hi, self.stride):
                if seq not in buf:
                    self._send_e2e_feedback(gid, FeedbackType.NACK, PacketID(gid, self.t), [seq])

    def _send_e2e_feedbacks_ack_only(self) -> None:
        """E2E_TIMEOUT: ACK-only end-to-end feedback (the external SR-ARQ baseline).

        The source recovers losses with its own retransmission timer, so the
        receiver never NACKs; it only ACKs every physical arrival this tick
        (seq-based) so the source learns which seqs reached the destination and
        stops resending them. Forward-only relays make the forward delay
        deterministic, so a seq that is never ACKed is exactly a lost one -- which
        the source's timer resends a full end-to-end RTT after its last send."""
        for gid, seq in self._e2e_arrivals:
            self._send_e2e_feedback(gid, FeedbackType.ACK, PacketID(gid, self.t), [seq])

    def _send_e2e_feedback(self, global_path_id, feedback_type, related_packet_id,
                           related_information_packets) -> None:
        """Emit one feedback packet on the chain's dedicated end-to-end channel with
        the full end-to-end one-way delay (mirrors mp_mh's per-global-path E2E)."""
        channel = self.e2e_feedback_channels[global_path_id]
        fb = FeedbackPacket(
            global_path_id=global_path_id,
            type=feedback_type,
            related_packet_id=related_packet_id,
            prop_time_left_in_channel=self.e2e_prop_delay,
            creation_time=self.t,
            related_information_packets=related_information_packets,
        )
        fb.record_arrival_at(channel.channel_name, self.t)
        # add_packets_to_channel resets prop_time_left to the channel delay
        # (= e2e_prop_delay), so the feedback carries the end-to-end delay.
        channel.add_packets_to_channel([fb], time=self.t)
        self.add_sent_feedback_packet_to_history(copy.copy(fb))

    def __repr__(self) -> str:
        s = super().__repr__()
        s += f"\n  chains delivered: { {g: self.chain_delivered_count[g] for g in sorted(self.chain_delivered_count)} }"
        s += f"\n  total in-order delivered: {len(self.information_packets_decoding_times)}"
        return s
