import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))

from mp_mh_network.Sender import GeneralSender, GeneralSenderPath
from mp_mh_network.Channels import Path
from mp_mh_network.Packet import RLNCPacket

from sr_arq.SRPacket import DataPacket


class SRSenderPath(GeneralSenderPath):
    """Sender-side path wrapper that, on top of the inherited feedback/epsilon
    machinery, remembers which seq was transmitted in each slot so a slot-based
    NACK (global_path_id, creation_time) can be resolved back to a seq."""

    def __init__(self, path: Path, my_sender: "SRSender", path_index: int, initial_epsilon: float = 0.0):
        super().__init__(path, my_sender, path_index, initial_epsilon=initial_epsilon)
        self.unit_name = f"{my_sender.unit_name}.SRSenderPath[{path_index}]"
        # creation_time (slot) -> seq sent on this path at that slot
        self.creation_time_to_seq: dict[int, int] = {}

    def record_sent_seq(self, creation_time: int, seq: int):
        self.creation_time_to_seq[creation_time] = seq

    def resolve_seq(self, creation_time: int) -> int | None:
        return self.creation_time_to_seq.get(creation_time)


class SRSender(GeneralSender):
    """Selective-Repeat ARQ source.

    - Single shared sequence stream striped across all P paths (no per-path
      pinning): a shared retransmit queue plus a monotonically increasing
      next-new-seq counter.
    - Lowest-seq-first scheduling: each free path takes the lowest pending seq
      from {retransmit queue} U {next new seq}. Since retransmits always have a
      lower seq than any not-yet-sent new seq, this drains retransmits first.
    - NACK-driven retransmission: a NACK is slot-based; it is mapped back to a
      seq via the path's slot->seq record and re-queued (unless already ACKed).
      ACKs carry the seq directly. No window, no timer.
    - Hop-by-hop: ACK/NACK come from the FIRST node, so 'delivered' here means
      'delivered to the next hop', not to the final receiver.
    """

    def __init__(
        self,
        num_of_packets_to_send: int,
        rtt: int,
        paths: list[Path],
        initial_epsilon: float = 0.0,
        window: int = None,
        next_hop=None,
        debug: bool = False,
    ):
        super().__init__(rtt, paths, init_paths=False, initial_epsilon=initial_epsilon, debug=debug)
        self.unit_name = "SRSender"
        self.num_of_packets_to_send = num_of_packets_to_send
        self.next_hop = next_hop

        # Sliding-window flow control (NOT reliability; reliability is NACK-driven).
        # A NEW seq may only be sent while it lies within [send_base, send_base+window),
        # where send_base is the lowest not-yet-ACKed seq. A stuck (repeatedly erased)
        # low seq therefore freezes the window and stalls new transmissions, which
        # is what caps SR-ARQ throughput below the raw link rate. window=None => no
        # window (unbounded; sender front-loads and approaches capacity).
        self.window: int | None = window
        self.send_base: int = 1  # lowest not-yet-ACKed seq

        # Wrap paths now that unit_name is set (GeneralSenderPath uses it).
        self.paths = [SRSenderPath(path, self, i, initial_epsilon) for i, path in enumerate(paths)]
        self.num_of_paths = len(self.paths)
        self.global_id_to_path: dict[int, SRSenderPath] = {
            p.get_global_path_index(): p for p in self.paths
        }

        # SR state
        self.next_new_seq: int = 1  # seqs are 1..num_of_packets_to_send
        self.retransmit_queue: set[int] = set()  # seqs awaiting retransmission
        self.acked_seqs: set[int] = set()  # seqs ACKed by the next hop

        # Stats attributes consumed by Network.collect_sender_stats / SimulationStats.
        # Every forward-channel transmission (new or retransmit, dropped or not)
        # is appended here so num_transmissions == n. FEC/FB-FEC stay empty.
        self.sent_new_rlnc_history: list[RLNCPacket] = []
        self.sent_fec_history: list[RLNCPacket] = []
        self.sent_fb_fec_history: list[RLNCPacket] = []

    def run_step(self, time: int = None):
        # Updates t and collects current feedbacks into self.feedbacks (trimmed
        # to slots not after the latest packet on air).
        super().run_step(time)
        self._process_feedbacks()
        self._transmit()
        # The SR network orchestrates the tick explicitly (next_hop=None); this
        # guarded cascade only fires if used in a simple single-hop wiring.
        if hasattr(self.next_hop, "run_step"):
            self.next_hop.run_step()

    def _process_feedbacks(self):
        # ACKs first: mark delivered and clear any pending retransmit.
        for fb in self.feedbacks:
            if fb.is_ack():
                for seq in (fb.get_related_information_packets() or []):
                    self.acked_seqs.add(seq)
                    self.retransmit_queue.discard(seq)
        # Advance the window base past all contiguously-ACKed seqs.
        self._advance_send_base()
        # Then NACKs: resolve slot -> seq and re-queue if not already delivered.
        for fb in self.feedbacks:
            if fb.is_nack():
                path = self.global_id_to_path.get(fb.get_global_path())
                if path is None:
                    continue
                creation_time = fb.get_related_packet_id().get_creation_time()
                seq = path.resolve_seq(creation_time)
                if seq is not None and seq not in self.acked_seqs:
                    self.retransmit_queue.add(seq)
                    self.sim_print(f"NACK -> re-queue seq {seq} (slot {creation_time}, path {path.get_global_path_index()})")

    def _advance_send_base(self):
        """Advance past the contiguous ACKed prefix of this sender's stream."""
        while self.send_base in self.acked_seqs:
            self.send_base += 1

    def _transmit(self):
        for path in self.paths:
            seq = self._next_seq_to_send()
            if seq is not None:
                self._send_seq_on_path(path, seq)
            else:
                # Nothing to send on this path: still advance its forward channel
                # so in-flight packets keep propagating.
                path.run_forward_channel_step(current_time=self.t)

    def _next_seq_to_send(self) -> int | None:
        # Retransmits always have a lower seq than the next new seq, so honoring
        # the retransmit queue first is exactly lowest-seq-first.
        if self.retransmit_queue:
            seq = min(self.retransmit_queue)
            self.retransmit_queue.discard(seq)
            return seq
        # New seq only if it is within the sliding window [send_base, send_base+window).
        if self.next_new_seq <= self.num_of_packets_to_send:
            if self.window is None or self.next_new_seq < self.send_base + self.window:
                seq = self.next_new_seq
                self.next_new_seq += 1
                return seq
        return None

    def _send_seq_on_path(self, path: SRSenderPath, seq: int):
        packet = DataPacket(
            global_path_id=path.get_global_path_index(),
            seq=seq,
            prop_time_left_in_channel=path.get_propagation_delay(),
            creation_time=self.t,
        )
        # send_packet adds to the forward channel, runs the forward step
        # (creation_time is set to self.t on transmission), calls
        # new_transmission_updates, and updates latest_rlnc_packet_on_air.
        self.send_packet(path, packet)
        path.record_sent_seq(self.t, seq)
        self.sim_print(f"sent seq {seq} on path {path.get_global_path_index()} at slot {self.t}")

    def new_transmission_updates(self, packet):
        # Record first-transmission time per seq (inherited) and count the
        # transmission toward n (every send, including retransmits/drops).
        super().new_transmission_updates(packet)
        self.sent_new_rlnc_history.append(packet)

    def __repr__(self) -> str:
        s = "SRSender:"
        s += f"\n  num_of_packets_to_send: {self.num_of_packets_to_send}"
        s += f"\n  rtt: {self.rtt}"
        s += f"\n  num paths: {self.num_of_paths}"
        s += f"\n  next_new_seq: {self.next_new_seq}"
        s += f"\n  num delivered (to next hop): {len(self.acked_seqs)}"
        s += f"\n  retransmit queue size: {len(self.retransmit_queue)}"
        s += f"\n  total transmissions: {len(self.sent_new_rlnc_history)}"
        return s


class SRSimSender(SRSender):
    """Per-path-independent SR ARQ source (the paper's baseline).

    Unlike SRSender (one shared stream striped+rerouted across paths), here each
    path runs its own independent SR ARQ on a STATIC, round-robin slice of the
    sequence space: path with list-index i owns seqs {i+1, i+1+P, i+1+2P, ...}.
    A dropped seq is retransmitted ONLY on its owning path (no rerouting), so a
    high-erasure path bottlenecks global in-order delivery. The receiver still
    reorders globally (same SRReceiver), so the metrics are comparable.
    """

    def __init__(self, *args, packets_per_path: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        P = self.num_of_paths
        # Equal per-path new-packet quota (None = unlimited). Caps admission of
        # new seqs only; retransmits are unaffected.
        if packets_per_path is not None:
            assert packets_per_path > 0, (
                f"packets_per_path must be > 0, got {packets_per_path}"
            )
        self.packets_per_path: int | None = packets_per_path
        self.path_admitted_count: dict[int, int] = {i: 0 for i in range(P)}
        # Per-path next new seq (round-robin slice) and per-path retransmit set.
        self.path_next_new_seq: dict[int, int] = {i: i + 1 for i in range(P)}
        self.path_retransmit: dict[int, set[int]] = {i: set() for i in range(P)}
        # Per-path sliding-window base: lowest not-yet-ACKed seq owned by path i.
        # Path i's slice strides by P, so the base advances by P. Same range-based
        # window as SRSender, applied independently per path/chain.
        self.path_send_base: dict[int, int] = {i: i + 1 for i in range(P)}
        self.gid_to_index: dict[int, int] = {
            p.get_global_path_index(): i for i, p in enumerate(self.paths)
        }

    def _process_feedbacks(self):
        # ACKs first: clear the owning path's retransmit set.
        for fb in self.feedbacks:
            if fb.is_ack():
                i = self.gid_to_index.get(fb.get_global_path())
                for seq in (fb.get_related_information_packets() or []):
                    self.acked_seqs.add(seq)
                    if i is not None:
                        self.path_retransmit[i].discard(seq)
        # Advance each path's window base past its contiguously-ACKed seqs (stride P).
        for i in range(self.num_of_paths):
            while self.path_send_base[i] in self.acked_seqs:
                self.path_send_base[i] += self.num_of_paths
        # NACKs: resolve slot -> seq and re-queue on the SAME path (no rerouting).
        for fb in self.feedbacks:
            if fb.is_nack():
                i = self.gid_to_index.get(fb.get_global_path())
                if i is None:
                    continue
                path = self.paths[i]
                creation_time = fb.get_related_packet_id().get_creation_time()
                seq = path.resolve_seq(creation_time)
                if seq is not None and seq not in self.acked_seqs:
                    self.path_retransmit[i].add(seq)

    def _transmit(self):
        for i, path in enumerate(self.paths):
            seq = self._next_seq_for_path(i)
            if seq is not None:
                self._send_seq_on_path(path, seq)
            else:
                path.run_forward_channel_step(current_time=self.t)

    def _next_seq_for_path(self, i: int) -> int | None:
        # Lowest-seq-first within this path's own slice (retransmits are lower
        # than its next new seq).
        if self.path_retransmit[i]:
            seq = min(self.path_retransmit[i])
            self.path_retransmit[i].discard(seq)
            return seq
        # Equal per-path quota: stop admitting new seqs once this chain has
        # used its allowance (retransmits above still allowed).
        if (
            self.packets_per_path is not None
            and self.path_admitted_count[i] >= self.packets_per_path
        ):
            return None
        # New seq only if within this path's window: at most `window` outstanding
        # packets, i.e. (next_new - send_base)/P < window (P = stride).
        if self.path_next_new_seq[i] <= self.num_of_packets_to_send:
            within_window = (
                self.window is None
                or (self.path_next_new_seq[i] - self.path_send_base[i]) < self.window * self.num_of_paths
            )
            if within_window:
                seq = self.path_next_new_seq[i]
                self.path_next_new_seq[i] += self.num_of_paths
                self.path_admitted_count[i] += 1
                return seq
        return None

    def __repr__(self) -> str:
        s = "SRSimSender:"
        s += f"\n  num_of_packets_to_send: {self.num_of_packets_to_send}"
        s += f"\n  packets_per_path: {self.packets_per_path}"
        s += f"\n  num paths: {self.num_of_paths}"
        s += f"\n  per-path admitted: {self.path_admitted_count}"
        s += f"\n  per-path next new seq: {self.path_next_new_seq}"
        s += f"\n  per-path retransmit sizes: {[len(self.path_retransmit[i]) for i in range(self.num_of_paths)]}"
        s += f"\n  total transmissions: {len(self.sent_new_rlnc_history)}"
        return s
