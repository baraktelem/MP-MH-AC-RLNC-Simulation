from enum import Enum
from typing import Optional
from dataclasses import dataclass

class RLNCType(Enum):
    """Type of RLNC packet: NEW (information), A-PRIORI-FEC, or FB-FEC"""
    NEW = 0  # NEW information packet
    FEC = 1  # A-priori FEC packet, created by the sender
    FB_FEC = 2  # Feedback-based FEC packet, created by the sender
    CORRECTION = 3  # Correction packet created by intermediate nodes

class FeedbackType(Enum):
    """Type of feedback: ACK (innovative) or NACK (non-innovative)"""
    ACK = 0  # Packet increased rank (innovative)
    NACK = 1  # Packet did not increase rank (non-innovative)

@dataclass(frozen=True)
class PacketID:
    global_path_id: int
    creation_time: int
    # type: RLNCType | FeedbackType

    def get_global_path_id(self) -> int:
        return self.global_path_id
    
    def get_creation_time(self) -> int:
        return self.creation_time
    
    # def get_type(self) -> RLNCType | FeedbackType:
    #     return self.type

class Packet:
    """Base class for all packets in the simulation"""
    
    def __init__(self, 
                 global_path_id: int,
                 prop_time_left_in_channel: int,
                 creation_time: int,
                 type: RLNCType|FeedbackType = None):
        self.global_path_id = global_path_id
        self.prop_time_left_in_channel = prop_time_left_in_channel
        self.creation_time = creation_time
        self.arrival_times: dict[str, int] = {}
        self.type = type
        # self.id = PacketID(global_path_id, creation_time, type)
        self.id = PacketID(global_path_id, creation_time)
    
    def record_arrival_at(self, component_name: str, arrival_time: int):
        """Record when this packet arrived at a specific component"""
        self.arrival_times[component_name] = arrival_time
    
    def update_prop_time_left_in_channel(self, propagation_delay: int):
        """Update the propagation time left in the channel"""
        self.prop_time_left_in_channel = propagation_delay

    def get_creation_time(self) -> int:
        return self.creation_time
    
    def set_creation_time(self, creation_time: int):
        self.creation_time = creation_time

    def get_prop_time_left_in_channel(self) -> int:
        return self.prop_time_left_in_channel

    def get_global_path(self) -> int:
        return self.global_path_id

    def get_id(self) -> PacketID:
        return self.id

    def __repr__(self):
        s = (f"Packet{self.creation_time}: ID={self.id}; "
             f"prop_time_left_in_channel={self.prop_time_left_in_channel}; "
             f"arrival_times={self.arrival_times})")
        return s


class RLNCPacket(Packet):
    """
    RLNC (Random Linear Network Coding) packet used in forward channels.
    
    Contains:
    - Path-id tag: which global path this packet belongs to
    - Generation index: which RTT-block/generation it belongs to
    - Packet type: NEW/FEC/FB-FEC flag for debugging and scheduling
    - Information packets: list of source packet indices included in this coded packet
    """
    
    def __init__(self,
                 global_path_id: int,
                 type: RLNCType,
                  information_packets: list[int],
                 prop_time_left_in_channel: int,
                 creation_time: int):
        super().__init__(global_path_id, prop_time_left_in_channel, creation_time, type)
        self.information_packets = information_packets.copy()  # Coefficients for RLNC

    def get_information_packets(self) -> list[int]:
        return self.information_packets.copy()
    
    def get_type(self) -> RLNCType:
        return self.type

    # def get_id(self) -> int:
    #     return super().get_creation_time()
    
    def __repr__(self):
        s = super().__repr__()
        return ("RLNC" + s + f"; info packets={self.information_packets} | ")


class FeedbackPacket(Packet):
    """
    Feedback packet used in feedback channels for ACK/NACK signaling.
    
    Sent by receiver (or intermediate nodes in hop-by-hop mode) to indicate
    whether a received packet was innovative (ACK) or redundant (NACK).
    """
    
    def __init__(self,
                 global_path_id: int,
                 type: FeedbackType,
                 related_packet_id: PacketID,
                 prop_time_left_in_channel: int,
                 creation_time: int,
                 related_information_packets: Optional[list[int]] = None):
        super().__init__(global_path_id, prop_time_left_in_channel, creation_time, type)
        self.related_packet_id = related_packet_id  # ID of packet this feedback refers to
        self.related_information_packets = related_information_packets.copy() if related_information_packets is not None else None
    
    def is_ack(self) -> bool:
        """Returns True if this is an ACK (innovative packet)"""
        return self.type == FeedbackType.ACK

    def get_related_information_packets(self) -> Optional[list[int]]:
        return self.related_information_packets.copy() if self.related_information_packets is not None else None
    
    def is_nack(self) -> bool:
        """Returns True if this is a NACK (non-innovative packet)"""
        return self.type == FeedbackType.NACK

    def get_related_packet_id(self) -> PacketID:
        return self.related_packet_id
    
    def get_type(self) -> FeedbackType:
        return self.type

    def __repr__(self):
        s = super().__repr__()
        return (f"{self.type.name}" + s + f"; related RLNC={self.get_related_packet_id()} ; related info packets={self.get_related_information_packets()} | ")