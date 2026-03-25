from Packet import Packet, RLNCPacket, FeedbackPacket, RLNCType, FeedbackType, PacketID
from Channels import Channel, ForwardChannel, Path
from CodedEquation import CodedEquation
from typing import Optional
import copy

class ReceiverPath(Path):
    def __init__(self, path: Path,
                path_index: int,
                my_receiver):
        self.__dict__.update(path.__dict__) # Automatically copy all attributes from the existing path
        
        # Add ReceiverPath-specific attributes
        self.my_receiver = my_receiver
        self.path_index = path_index
        self.received_packets = [] # All packets received through simulation
        self.receiving_packets_strating_time = None
        self.unit_name = f"{self.my_receiver.unit_name}.ReceiverPath[{path_index}]"

    def received_packets_is_empty(self) -> bool:
        return len(self.received_packets) == 0

    def pop_arrived_packets(self):
        return self.forward_channel.pop_arrived_packets()

    def get_receiving_packets_strating_time(self) -> int:
        return self.receiving_packets_strating_time

    def get_global_path_index(self):
        return self.global_path_index

    def get_received_channel_history(self):
        return self.forward_channel.get_channel_history()

    def get_sent_feedback_channel_history(self):
        return self.feedback_channel.get_channel_history()

    def send_feedback_packet(self, feedback_packet: FeedbackPacket, current_time: int) -> FeedbackPacket:
        feedback_packet = super().send_feedback_packet(feedback_packet, current_time)
        if feedback_packet is not None:
            self.sim_print(f"Feedback packet sent:\n\t{feedback_packet}")
            self.my_receiver.add_sent_feedback_packet_to_history(copy.deepcopy(feedback_packet))
        return feedback_packet

    def update_receiving_packets_strating_time(self, arrived_packet: RLNCPacket, time: int):
        # Update starting time for first packet arrival
        if self.receiving_packets_strating_time is None:
            self.receiving_packets_strating_time = time
        # Handle case were first packet sent was lost
        elif self.receiving_packets_strating_time > arrived_packet.get_creation_time() + self.get_propagation_delay():
            # If first packet sent was lost, update starting time to the time the packet was supposed to be received
            self.receiving_packets_strating_time = arrived_packet.get_creation_time() + self.get_propagation_delay()

    def sim_print(self, message: str, time: int=None):
        t = time if time is not None else self.my_receiver.t
        super().sim_print(f"Path {self.path_index}: {message}", t)


class GeneralReceiver:
    def __init__(self,
                input_paths: list[Path],
                rtt: int,
                unit_name: str=None):
        self.unit_name = unit_name if unit_name is not None else "GeneralReceiver"

        # Receiver paths
        self.receiver_paths = [ReceiverPath(path, i, self) for i, path in enumerate(input_paths)]
        self.num_of_input_paths = len(input_paths)
        self.paths_propagation_time = rtt / 2 # Propagation time is half of the RTT
        assert len([path for path in input_paths if path.get_propagation_delay() != self.paths_propagation_time]) == 0, \
            "All paths must have the same propagation time"

        # Parameters
        self.rtt = rtt
        self.t = 0 # Current time step

        # Forward arrival from the input path served this step (round-robin); None if that path had nothing
        self.arrived_packet: Packet | None = None
        
        # Statistics
        self.received_rlnc_channel_history : list[RLNCPacket] = [] # All received RLNC packets
        self.sent_feedback_channel_history : list[FeedbackPacket] = [] # All sent feedback packets

    def run_step(self, time: int=None):
        # Update time
        if time is not None:
            self.t = time
        else:
            self.t += 1
        self.arrived_packet = None
        for receiver_path in self.receiver_paths:
            # Get new packet from path's forward channel
            arrived_packets = receiver_path.pop_arrived_packets()
            assert arrived_packets is None or len(arrived_packets) <= 1, \
                f"Only 1 packet can be in receiver path buffer, but {len(arrived_packets)} were found on receiver path {receiver_path.path_index}\n\t{arrived_packets}"
            # Send nack if no packets were received
            self.sim_print(f"Creating feedback packets for path {receiver_path.path_index}: {arrived_packets}")
            if arrived_packets is None or arrived_packets == []:
                self.send_nack(receiver_path)
            else:
                self.arrived_packet = arrived_packets.pop(0) # Arrived packet is a list with len=1, so we pop the first element
                # Add packet to history
                self.received_rlnc_channel_history.append(self.arrived_packet)
                # Update starting time according to arrived packets
                receiver_path.update_receiving_packets_strating_time(self.arrived_packet, self.t)
                # Send ack
                self.send_ack(receiver_path, self.arrived_packet)
                self._after_rlnc_arrived(receiver_path, self.arrived_packet)

    def _after_rlnc_arrived(self, receiver_path: ReceiverPath, arrived_packet: RLNCPacket) -> None:
        """Subclass hook after an RLNC packet is taken from a path (like decode for SimReceiver)."""
        pass

    def send_nack(self, receiver_path: ReceiverPath):
        if self.t > self.paths_propagation_time: # Wait for first packet to arrive
            # Calculate when the missing packet should have been sent
            expected_packet_creation_time = self.t - self.paths_propagation_time
            self.sim_print(f"Sending NACK for C{expected_packet_creation_time} on path {receiver_path.path_index}")
            
            # Create PacketID for the missing packet (we don't know the actual type, assume NEW)
            related_packet_id = PacketID(
                global_path_id=receiver_path.get_global_path_index(),
                creation_time=expected_packet_creation_time,
                # type=RLNCType.NEW  # Assume NEW type for missing packets
            )
            
            nack_packet = FeedbackPacket(
                global_path_id=receiver_path.get_global_path_index(),
                type=FeedbackType.NACK,
                related_packet_id=related_packet_id,
                prop_time_left_in_channel=receiver_path.get_propagation_delay(),
                creation_time=self.t,
                related_information_packets=None
            )
            receiver_path.send_feedback_packet(nack_packet, current_time=self.t)
        else:
            self.sim_print(f"Not sending NACK for path {receiver_path.path_index}- no packets received yet")

    def send_ack(self, receiver_path: ReceiverPath, arrived_packet: RLNCPacket):
        # Add ACK to pending buffer
        # self.sim_print(f"sending ACK for C{arrived_packet.get_creation_time()} on path {receiver_path.path_index}")
        self.sim_print(f"sending ACK for Packet: {arrived_packet.get_id()}")
        ack_packet = FeedbackPacket(
            global_path_id=receiver_path.get_global_path_index(),
            type=FeedbackType.ACK,
            related_packet_id=arrived_packet.get_id(),
            prop_time_left_in_channel=receiver_path.get_propagation_delay(),
            creation_time=self.t, # Will be set when the packet is sent
            related_information_packets=arrived_packet.get_information_packets()
        )
        receiver_path.send_feedback_packet(ack_packet, current_time=self.t)

    def get_received_rlnc_channel_history(self):
        return self.received_rlnc_channel_history

    def add_sent_feedback_packet_to_history(self, feedback_packet: FeedbackPacket):
        self.sent_feedback_channel_history.append(feedback_packet)

    def get_sent_feedback_channel_history(self) -> list[FeedbackPacket]:
        return self.sent_feedback_channel_history

    def sim_print(self, message: str) -> None:
        print(f"[{self.t}] {self.unit_name}: {message}")

    def __repr__(self) -> str:
        s = f"{self.unit_name}:"
        s += f"\n  num paths: {len(self.receiver_paths)}"
        s += f"\n  num received RLNC packets: {len(self.received_rlnc_channel_history)}"
        s += f"\n  num sent feedback packets: {len(self.sent_feedback_channel_history)}"
        return s

    def get_receiver_path(self, path_index: int) -> ReceiverPath:
        return self.receiver_paths[path_index]


class SimReceiver(GeneralReceiver):
    def __init__(self,
                input_paths: list[Path],
                rtt: int,
                unit_name: str=None):
        # Set unit name before calling super() for setting name that is not "GeneralReceiver"
        if unit_name is None:
            unit_name = "SimReceiver"
        super().__init__(input_paths, rtt, unit_name)

        # Decoding
        self.coded_equations : list[CodedEquation] = [] # All undecoded equations
        self.coded_information_packets : set[int] = set() # All information packets that are still coded
        self.latest_decoded_information_packet = 0 # Latest decoded information packet

        # Statistics
        self.information_packets_decoding_times : dict[int, int] = {} # Mapping for each information packet to the time it was decoded

    def _after_rlnc_arrived(self, receiver_path: ReceiverPath, arrived_packet: RLNCPacket) -> None:
        self.decode_packets(arrived_packet)

    def update_information_packets_decode_times(self, information_packets: list[int]):
        self.sim_print(f"update_information_packets_decode_times: Updating information packets decoding times for packets:\n\t{information_packets}")
        for packet in information_packets:
            if packet not in self.information_packets_decoding_times:
                self.information_packets_decoding_times[packet] = self.t

    def decode_packets(self, arrived_packet: RLNCPacket):
        self.sim_print(f"decode_packets: Trying to decode packets:\n\t{arrived_packet.get_information_packets()}")
        # Get new information packets from received RLNC
        coded_information_packets_from_rlnc_packet = [p for p in arrived_packet.get_information_packets() if p > self.latest_decoded_information_packet]
        self.sim_print(f"decode_packets: Coded information packets from RLNC packet:\n\t{coded_information_packets_from_rlnc_packet}")
        if len(coded_information_packets_from_rlnc_packet) == 0: # Ignore packets after all information packets were decoded
            return
        # Add new information packets to equations
        new_coded_equation = CodedEquation(arrived_packet.get_id(), coded_information_packets_from_rlnc_packet)
        self.coded_equations.append(new_coded_equation)
        # Update information packets that are still coded (unknowns)
        self.coded_information_packets.update(coded_information_packets_from_rlnc_packet)
        # Decode if #unknowns <= #equations
        if len(self.coded_information_packets) <= len(self.coded_equations):
            # All coded packets can now be decoded!
            packets_to_decode = list(self.coded_information_packets)
            self.sim_print(f"decode_packets: Successfully decoded ALL information packets:\n\t{packets_to_decode}")
            # Update decoded information packets
            self.latest_decoded_information_packet = max(self.coded_information_packets)
            self.update_information_packets_decode_times(packets_to_decode)  # Mark ALL coded packets as decoded
            # No unknowns left, clear all equations and information packets
            self.coded_equations.clear() # Clear all equations
            self.coded_information_packets.clear() # Clear all information packets

    def __repr__(self) -> str:
        s = super().__repr__()
        s += f"\n  latest decoded information packet: {self.latest_decoded_information_packet}"
        if hasattr(self, 't') and self.t > 0:
            total_information_decoded = len(self.information_packets_decoding_times)
            s += f"\n  normalized throughput for t{self.t}: {total_information_decoded / self.t}"


class NodeReceiver(GeneralReceiver):
    def __init__(
        self,
        hop_num: int,
        input_paths: list[Path],
        rtt: int,
        unit_name: str=None,
        parent_node: 'Node'=None,
    ):
        # Set unit name before calling super() for setting name that is not "GeneralReceiver"
        if unit_name is None:
            unit_name = f"NodeReceiver[{hop_num}]"
        super().__init__(input_paths, rtt, unit_name)
        self.hop_num = hop_num
        self.parent_node = parent_node

        # Buffers
        """Nodes are suppose to send linear combinations of either New RLNCs or Correction packets.
        Since that in the simulation we are not actually coding information packets, there is no point of sending those combinations.
        Hence, a Node will just send the all information packets in its buffer (which the receiver still hasn't decoded)"""
        self.new_information_packets_buffer : set[int] = set() # New RLNCs
        self.correction_information_packets_buffer : set[int] = set() # both FEC and FB-FEC

        # Statistics
        self.new_rlnc_packets_history : list[RLNCPacket] = []
        self.correction_packets_history : list[RLNCPacket] = []

    def _after_rlnc_arrived(self, receiver_path: ReceiverPath, arrived_packet: RLNCPacket) -> None:
        information_packets = set(arrived_packet.get_information_packets())
        if arrived_packet.get_type() == RLNCType.NEW:
            self.new_rlnc_packets_history.append(arrived_packet)
            self.new_information_packets_buffer.update(information_packets)
        else: # Correction packet - both FEC and FB-FEC
            self.correction_packets_history.append(arrived_packet)
            self.correction_information_packets_buffer.update(information_packets)

    def __repr__(self) -> str:
        s = super().__repr__()
        s += f"\n  hop number: {self.hop_num}"
        return s