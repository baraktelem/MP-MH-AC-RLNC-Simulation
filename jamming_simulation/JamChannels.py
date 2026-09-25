import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mp_mh_network.Channels import Path, ForwardChannel
from mp_mh_network.Packet import Packet, RLNCPacket

class JamPath(Path):
    def __init__(
        self, 
        propagation_delay: int, 
        epsilon: float, 
        hop_index: int, 
        path_index_in_hop: int, 
        name_prefix: str="", 
        debug: bool = False,
        store_history: bool = True
        ):
        super().__init__(propagation_delay, epsilon, hop_index, path_index_in_hop, name_prefix, debug, store_history=store_history)
        self.forward_channel = JamForwardChannel.from_forward_channel(self.forward_channel)

    @property
    def is_jammed(self) -> bool:
        return self.forward_channel.is_jammed

    def jam_path(self):
        self.forward_channel.is_jammed = True

    def unjam_path(self):
        self.forward_channel.is_jammed = False


class JamForwardChannel(ForwardChannel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_jammed: bool = False
        self.jammed_packets_history: list[RLNCPacket] = []
        self.num_jammed: int = 0  # always-on count; jammed_packets_history skipped in low-memory mode

    @classmethod
    def from_forward_channel(cls, forward_channel: ForwardChannel) -> "JamForwardChannel":
        forward_channel.__class__ = cls
        forward_channel.is_jammed = False
        forward_channel.jammed_packets_history = []
        forward_channel.num_jammed = 0
        # store_history is already set on the wrapped ForwardChannel instance and is
        # preserved across the __class__ reassignment above.
        return forward_channel

    def apply_noise_on_single_packet(self, packet: RLNCPacket) -> tuple[RLNCPacket, bool]:
        if self.is_jammed:
            self.num_jammed += 1
            if self.store_history:
                self.jammed_packets_history.append(packet)
            return packet, True
        return super().apply_noise_on_single_packet(packet)
