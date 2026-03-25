from Packet import PacketID


class CodedEquation:
    def __init__(self, related_rlnc_packet_id: PacketID, unknown_packets: list[int]):
        self.related_rlnc_packet_id = related_rlnc_packet_id # ID of the RLNC packet that with the coefficients of the equation
        self.unknown_packets = unknown_packets # The set of packets that are unknown in the equation

    def get_related_rlnc_packet_id(self):
        return self.related_rlnc_packet_id

    def get_unknown_packets(self):
        return self.unknown_packets

    def __repr__(self):
        return f"Equation{self.related_rlnc_packet_id}: Unknown packets: {self.unknown_packets}" 