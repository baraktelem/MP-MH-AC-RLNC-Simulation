from Packet import Packet, RLNCPacket, FeedbackPacket, RLNCType, NodeRLNCType, FeedbackType, PacketID
from Channels import Path, Channel
from CodedEquation import CodedEquation
from feedback_source import FeedbackSource
# from typing import Optional
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
            self.my_receiver.add_sent_feedback_packet_to_history(copy.copy(feedback_packet))
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
                hop_rtt: int,
                unit_name: str=None,
                debug: bool = False):
        self.unit_name = unit_name if unit_name is not None else "GeneralReceiver"
        self.debug = debug

        # Receiver paths
        self.receiver_paths = [ReceiverPath(path, i, self) for i, path in enumerate(input_paths)]
        self.num_of_input_paths = len(input_paths)
        self.paths_propagation_time = hop_rtt / 2 # Per-hop one-way delay is half of the per-hop RTT
        assert len([path for path in input_paths if path.get_propagation_delay() != self.paths_propagation_time]) == 0, \
            "All paths must have the same propagation time"

        # Parameters
        self.hop_rtt = hop_rtt
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
        if not self.debug:
            return
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
                hop_rtt: int,
                unit_name: str=None,
                debug: bool = False,
                feedback_source: FeedbackSource = FeedbackSource.HBH,
                e2e_feedback_channels: dict[int, Channel] = None,
                e2e_prop_delay: int = None):
        # Set unit name before calling super() for setting name that is not "GeneralReceiver"
        if unit_name is None:
            unit_name = "SimReceiver"
        super().__init__(input_paths, hop_rtt, unit_name, debug=debug)

        # End-to-end feedback wiring
        self.feedback_source = feedback_source
        assert feedback_source in (FeedbackSource.HBH, FeedbackSource.E2E), \
            f"Invalid feedback source: {feedback_source}"
        if feedback_source == FeedbackSource.E2E:
            assert e2e_feedback_channels is not None, \
                "E2E feedback channels are required for end-to-end feedback"
            assert e2e_prop_delay is not None, \
                "E2E propagation delay (global one-way delay) is required for end-to-end feedback"
        self.e2e_feedback_channels = e2e_feedback_channels
        self.e2e_prop_delay = e2e_prop_delay  # global (end-to-end) one-way delay = global_rtt / 2

        # Decoding
        self.coded_equations : list[CodedEquation] = [] # All undecoded equations
        self.coded_information_packets : set[int] = set() # All information packets that are still coded
        self.latest_decoded_information_packet = 0 # Latest decoded information packet

        # Statistics
        self.information_packets_decoding_times : dict[int, int] = {} # Mapping for each information packet to the time it was decoded

        # E2E feedback: packets that arrived this tick (carried global label is
        # authoritative). Collected during run_step, then turned into exactly one
        # end-to-end feedback per global path in _send_e2e_feedbacks.
        self._e2e_arrivals_this_tick : list[RLNCPacket] = []

    def run_step(self, time: int=None):
        # Base step: per-hop feedback (to the last node, HBH and E2E alike) plus
        # decoding. Arrivals are captured in _after_rlnc_arrived below.
        self._e2e_arrivals_this_tick = []
        super().run_step(time)
        # End-to-end feedback to the source is emitted as a label-complete pass
        # once all arrivals for this tick are known (see _send_e2e_feedbacks).
        if self.feedback_source == FeedbackSource.E2E:
            self._send_e2e_feedbacks()

    def _after_rlnc_arrived(self, receiver_path: ReceiverPath, arrived_packet: RLNCPacket) -> None:
        if self.feedback_source == FeedbackSource.E2E:
            self._e2e_arrivals_this_tick.append(arrived_packet)
        self.decode_packets(arrived_packet)

    def _send_e2e_feedbacks(self) -> None:
        """Send exactly one end-to-end feedback *per global path* for this tick.

        Per-hop feedback to the last node is handled by the base send_ack/send_nack
        during super().run_step(); this pass is only the end-to-end feedback to the
        source. It must be label-complete: because intermediate nodes (including
        the last one) re-run natural matching every tick, a packet's carried global
        label comes from the matching in force when it was *forwarded*, which can
        differ from the physical path's current label. All packets arriving at the
        receiver in a single tick were forwarded by the last node at the same time,
        so their carried labels form a subset of one bijection over {1..P} and are
        therefore distinct. We ACK those carried labels, then NACK every remaining
        global path (the labels erased on the last hop). Doing it per physical empty
        slot instead would mismatch the carried labels and produce duplicate/missing
        feedback for the same (label, creation_time).

        Every arrival is ACKed: intermediate nodes recode (they re-send from their
        correction buffer to fill slots left empty by upstream erasures) instead of
        forwarding a per-hop DROPPED marker, so an arrival is always a genuine
        delivery on that global path. Consequently the only end-to-end NACKs are for
        labels erased on the last hop, which keeps each global path at its min-cut
        (bottleneck-hop) rate rather than the product of the per-hop erasures."""
        # Warm-up: before the first end-to-end packet could have arrived there is
        # nothing to ACK and we must not invent NACKs for not-yet-flowing labels.
        if self.t <= self.e2e_prop_delay:
            return

        creation_time = self.t - self.e2e_prop_delay
        arrived_global_paths : set[int] = set()

        # ACK every arrival (see docstring: recoding makes each arrival a genuine
        # delivery on its carried global path).
        for arrived_packet in self._e2e_arrivals_this_tick:
            global_path_id = arrived_packet.get_global_path()
            arrived_global_paths.add(global_path_id)
            related_packet_id = PacketID(
                global_path_id=global_path_id,
                creation_time=creation_time,
            )
            self.sim_print(f"E2E: sending ACK for {related_packet_id}")
            self._send_e2e_feedback_on_a_channel(
                global_path_id, FeedbackType.ACK, related_packet_id,
                arrived_packet.get_information_packets(),
            )

        # NACK every global path that did not arrive this tick (erased on the last hop).
        for global_path_id in self.e2e_feedback_channels.keys():
            if global_path_id in arrived_global_paths:
                continue
            related_packet_id = PacketID(
                global_path_id=global_path_id,
                creation_time=creation_time,
            )
            self.sim_print(f"E2E: sending NACK (missing label) for {related_packet_id}")
            self._send_e2e_feedback_on_a_channel(global_path_id, FeedbackType.NACK, related_packet_id, None)

    def _send_e2e_feedback_on_a_channel(self,
                           global_path_id: int,
                           feedback_type: FeedbackType,
                           related_packet_id: PacketID,
                           related_information_packets) -> None:
        """Emit a feedback packet onto the dedicated end-to-end channel for the
        given global path, with the full end-to-end one-way delay, and keep the
        sent-feedback history bookkeeping consistent with the HBH path."""
        channel = self.e2e_feedback_channels[global_path_id]
        feedback_packet = FeedbackPacket(
            global_path_id=global_path_id,
            type=feedback_type,
            related_packet_id=related_packet_id,
            prop_time_left_in_channel=self.e2e_prop_delay,
            creation_time=self.t,
            related_information_packets=related_information_packets,
        )
        feedback_packet.record_arrival_at(channel.channel_name, self.t)
        # add_packets_to_channel resets prop_time_left to the channel's delay
        # (= global_prop_delay), so the feedback carries the end-to-end delay.
        channel.add_packets_to_channel([feedback_packet], time=self.t)
        self.add_sent_feedback_packet_to_history(copy.copy(feedback_packet))

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
        return s


class NodeReceiver(GeneralReceiver):
    def __init__(
        self,
        hop_num: int,
        input_paths: list[Path],
        hop_rtt: int,
        unit_name: str=None,
        parent_node: 'Node'=None,
        debug: bool = False,
    ):
        # Constants
        if unit_name is None:   # Set unit name before calling super() for setting name that is not "GeneralReceiver"
            unit_name = f"NodeReceiver[{hop_num}]"
        super().__init__(input_paths, hop_rtt, unit_name, debug=debug)
        self.hop_num = hop_num

        # Network
        self.parent_node = parent_node

        # Buffers
        """Nodes are suppose to send linear combinations of either New RLNCs or Correction packets.
        Since that in the simulation we are not actually coding information packets, there is no point of sending those combinations.
        Hence, a Node will just send the all information packets in its buffer (which the receiver still hasn't decoded)"""
        self.new_information_packets_buffer : set[int] = set() # New RLNCs
        self.correction_information_packets_buffer : set[int] = set() # both FEC and FB-FEC

        # Natural Matching Tracking
        self.global_paths_rlnc_types : dict[int, NodeRLNCType | RLNCType] = {} # Mapping for each global path index to the RLNC type

        # Statistics
        self.new_rlnc_packets_history : list[RLNCPacket] = []
        self.correction_packets_history : list[RLNCPacket] = []

    def run_step(self, time: int=None):
        self.global_paths_rlnc_types = {}
        super().run_step(time)
        # Mark all dropped packets in mapping
        for path in self.receiver_paths: 
            global_path_idx = path.get_global_path_index()
            if self.global_paths_rlnc_types.get(global_path_idx, None) is None:
                self.global_paths_rlnc_types[global_path_idx] = NodeRLNCType.DROPPED
    
    def _after_rlnc_arrived(self, receiver_path: ReceiverPath, arrived_packet: RLNCPacket) -> None:
        """This function is called for each RLNC that arrives on a path. It does the following:
        1. Add packet to buffer
        2. Add packet to history
        3. Map packet type to global paths"""
        # TODO: Maybe on first packet arrival we should update both buffers
        information_packets = set(arrived_packet.get_information_packets())
        if arrived_packet.get_type() == RLNCType.NEW:
            self.new_information_packets_buffer.update(information_packets)
            self.new_rlnc_packets_history.append(arrived_packet)
        else: # Correction packet - FEC, FB-FEC or CORRECTION
            self.correction_information_packets_buffer.update(information_packets)
            self.correction_packets_history.append(arrived_packet)
        
        # Assert each global path is mapped only once at each time step
        global_path_id = arrived_packet.get_global_path()
        assert self.global_paths_rlnc_types.get(global_path_id, None) is None, \
            f"[{self.t}] {self.unit_name}: Packet type already mapped to global path {global_path_id}"
        
        # Map packet type to global paths
        self.global_paths_rlnc_types[global_path_id] = arrived_packet.get_type()

    def get_global_paths_rlnc_types(self) -> dict[int, NodeRLNCType | RLNCType]:
        return self.global_paths_rlnc_types

    def __repr__(self) -> str:
        s = super().__repr__()
        s += f"\n  hop number: {self.hop_num}"
        return s