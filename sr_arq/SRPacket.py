import sys
import os
from enum import auto

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))

from mp_mh_network.Packet import Packet, PacketType


class SRType(PacketType):
    """Type tag for uncoded Selective-Repeat ARQ data packets."""
    DATA = auto()


class DataPacket(Packet):
    """Uncoded SR ARQ data packet.

    Unlike RLNCPacket it carries exactly one information packet (its sequence
    number). It still exposes get_information_packets() -> [seq] so the existing
    GeneralReceiver feedback logic and Network stats pipeline work unchanged.
    """

    def __init__(
        self,
        global_path_id: int,
        seq: int,
        prop_time_left_in_channel: int,
        creation_time: int,
    ):
        super().__init__(global_path_id, prop_time_left_in_channel, creation_time, type=SRType.DATA)
        self.seq = seq

    def get_seq(self) -> int:
        return self.seq

    def get_information_packets(self) -> list[int]:
        return [self.seq]

    def get_type(self) -> SRType:
        return self.type

    def __repr__(self) -> str:
        s = super().__repr__()
        return "Data" + s + f"; seq={self.seq} | "
