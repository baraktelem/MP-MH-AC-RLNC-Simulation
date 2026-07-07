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
