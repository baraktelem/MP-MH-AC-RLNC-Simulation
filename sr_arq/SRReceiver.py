import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))

from mp_mh_network.Receiver import GeneralReceiver, ReceiverPath
from mp_mh_network.Packet import RLNCPacket
from mp_mh_network.Channels import Path


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

    def __init__(self, input_paths: list[Path], rtt: int, num_chains: int, unit_name: str = None, debug: bool = False):
        if unit_name is None:
            unit_name = "SRSimReceiver"
        super().__init__(input_paths, rtt, unit_name, debug=debug)
        self.stride = num_chains  # global round-robin stride (= P)

        # Per-chain in-order state (chain id = global_path_id = c+1).
        self.expected: dict[int, int] = {}       # gid -> next in-order seq for that chain
        self.buffer: dict[int, set[int]] = {}    # gid -> arrived but not yet in-order seqs

        # Stats.
        self.information_packets_decoding_times: dict[int, int] = {}  # global seq -> delivery time
        self.chain_finish_time: dict[int, int] = {}                   # gid -> last delivery time
        self.chain_delivered_count: dict[int, int] = {}               # gid -> #delivered

    def _after_rlnc_arrived(self, receiver_path: ReceiverPath, arrived_packet: RLNCPacket) -> None:
        gid = arrived_packet.get_global_path()
        seq = arrived_packet.get_information_packets()[0]
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

    def __repr__(self) -> str:
        s = super().__repr__()
        s += f"\n  chains delivered: { {g: self.chain_delivered_count[g] for g in sorted(self.chain_delivered_count)} }"
        s += f"\n  total in-order delivered: {len(self.information_packets_decoding_times)}"
        return s
