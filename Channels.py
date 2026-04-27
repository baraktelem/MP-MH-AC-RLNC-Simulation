import random
from copy import deepcopy
from Packet import Packet, RLNCPacket, RLNCType, PacketID, FeedbackPacket

class Path:
    global_path_index: int
    hop_index: int
    path_index_in_hop: int # Index of the path in the sender side of the hop
    my_sender: None # The sender object that is responsible for sending packets on this path- can be either Sender or Node
    my_receiver: None # The receiver object that is responsible for receiving packets on this path- can be either Receiver or Node
    forward_channel: 'ForwardChannel' # The channel object that is responsible for forwarding packets on this path
    feedback_channel: 'Channel' # The channel object that is responsible for sending ACK/NACKs back to the sender
    propagation_delay: int

    def __init__(self, propagation_delay: int, epsilon: float, hop_index: int, path_index_in_hop: int, name_prefix: str=""):
        self.hop_index = hop_index
        self.path_index_in_hop = path_index_in_hop
        self.unit_name = f"{name_prefix}Path[{hop_index}][{path_index_in_hop}]"
        self.forward_channel = ForwardChannel(propagation_delay, epsilon, hop_index, path_index_in_hop, name_prefix=self.unit_name+".")
        self.feedback_channel = Channel(propagation_delay, hop_index, path_index_in_hop, name_prefix=self.unit_name+".Feedback")
        self.my_sender = None
        self.my_receiver = None
        assert self.feedback_channel.get_propagation_delay() == self.forward_channel.get_propagation_delay(), \
            "Feedback channel must have the same propagation delay as the forward channel"
        self.propagation_delay = self.forward_channel.get_propagation_delay()

    def send_feedback_packet(self, packet: Packet, current_time: int) -> Packet:
        packet.record_arrival_at(self.unit_name, current_time)
        self.sim_print(f"Adding packet to feedback channel:\n\t{packet}", current_time)
        self.feedback_channel.add_packets_to_channel([packet], time=current_time)
        return packet

    def get_pending_packets_buffer(self):
        return self.forward_channel.get_pending_packets_buffer()

    def set_global_path_index(self, global_path_index: int):
        self.global_path_index = global_path_index
        self.forward_channel.set_global_path_index(global_path_index)
        self.feedback_channel.set_global_path_index(global_path_index)

    def get_global_path_index(self):
        return self.global_path_index
    
    def __repr__(self):
        s = f"Path(hop_index={self.hop_index}, path_index_in_hop={self.path_index_in_hop}, global_path_index={self.global_path_index})"
        s += f"\n  Forward Channel:\n    {self.forward_channel}"
        s += f"\n  Feedback Channel:\n    {self.feedback_channel}"
        return s

    def is_forward_channel_empty(self):
        return self.forward_channel.is_empty()

    def get_propagation_delay(self):
        return self.propagation_delay

    def get_forward_channel_history(self, include_dropped_packets: bool = False) -> list[RLNCPacket]:
        return self.forward_channel.get_channel_history(include_dropped_packets=include_dropped_packets)
    
    def get_feedback_channel_history(self) -> list[FeedbackPacket]:
        return self.feedback_channel.get_channel_history()

    def add_packet_to_forward_channel(self, packet: Packet, current_time: int):
        self.forward_channel.add_packets_to_pending_packets_buffer([packet], current_time=current_time)

    # def get_pending_packets_buffer(self):
    #     return self.pending_packets_buffer
    
    def get_dropped_packets(self):
        return self.forward_channel.get_dropped_packets()

    def run_forward_channel_step(self, current_time: int) -> RLNCPacket | None:
        rlnc_packet = self.forward_channel.run_step(current_time=current_time)
        return rlnc_packet

    def sim_print(self, message: str, time: int=None):
        if time is not None:
            print(f"[{time}] {self.unit_name}: {message}")
        else:
            print(f"    {self.unit_name}: {message}")

class Channel:
    propagation_delay: int  # Time that takes a packet to travel through the channel
    packets_in_channel: list[Packet]  # Packets currently propagating through the channel
    arrived_packets: list[Packet] # Packets that have finished propagating through the channel (used as a queue)
    # Packets that have just entered the channel from the sender, waiting to start propagation.
    # These packets will not have their propagation_delay decremented during the first run_step()
    pending_packets: list[Packet] 
    hop_index: int
    path_index_in_hop: int
    global_path_index: int
    channel_name: str
    
    def __init__(self, propagation_delay: int, hop_index: int, path_index_in_hop: int, name_prefix: str=""):
        self.propagation_delay = propagation_delay
        self.packets_in_channel = []
        self.arrived_packets = []
        self.hop_index = hop_index
        self.path_index_in_hop = path_index_in_hop
        self.channel_name = f"{name_prefix}Channel[{hop_index}][{path_index_in_hop}]"
        self.channel_history = [] # All packets that have passed through the channel

    def run_step(self):
        """ Decrement propagation time of all packets in channel and separate arrived from in-transit """
        # Decrement all packet times and separate arrived from in-transit
        packets_still_in_channel = []
        packets_arrived = []
        for packet in self.packets_in_channel:
            packet.prop_time_left_in_channel -= 1
            if packet.prop_time_left_in_channel > 0:
                packets_still_in_channel.append(packet)
            else:
                packets_arrived.append(packet)
        
        # Update channel to only contain packets still in transit
        self.packets_in_channel = packets_still_in_channel
        
        # Add arrived packets to the queue and verify they all have prop_time == 0
        for packet in packets_arrived:
            assert packet.prop_time_left_in_channel == 0, \
                f"{self.channel_name}: Arrived packet has prop_time_left_in_channel={packet.prop_time_left_in_channel}, expected 0:\n    {packet}"
            self.arrived_packets.append(packet)
        print(f"[{self.channel_name}]: All arrived packets:\n\t{packets_arrived}")
        print(f"[{self.channel_name}]: All packets still in channel:\n\t{self.packets_in_channel}")
    
    def add_packet_to_history(self, packet):
        self.channel_history.append(deepcopy(packet))

    def get_channel_history(self) -> list[Packet]:
        return self.channel_history
        
    def pop_arrived_packets(self):
        """Return all packets that have finished propagating through the channel"""
        arrived_packets = self.arrived_packets
        # Assert all arrived packets have prop_time == 0
        for packet in arrived_packets:
            assert packet.prop_time_left_in_channel == 0, \
                f"{self.channel_name}: Arrived packet has prop_time_left_in_channel={packet.prop_time_left_in_channel}, expected 0:\n    {packet}" 
        self.arrived_packets = []
        return arrived_packets if len(arrived_packets) > 0 else None

    def set_global_path_index(self, global_path_index: int):
        self.global_path_index = global_path_index

    def add_packets_to_channel(self, packets: list[Packet], time: int):
        """Add packets to channel"""
        self.sim_print(f"Adding packets to channel:\n\t{packets}", time)
        for packet in packets:
            packet.update_prop_time_left_in_channel(self.propagation_delay)
            packet.record_arrival_at(self.channel_name, time)
            self.packets_in_channel.append(packet)
            self.add_packet_to_history(packet)
        self.sim_print(f"Packets added to channel:\n\t{packets}", time)

    def is_empty(self):
        """Check if channel has any packets"""
        return  len(self.packets_in_channel) == 0

    def get_propagation_delay(self):
        return self.propagation_delay

    # def get_pending_packets_buffer(self):
    #     return self.pending_packets_buffer

    def __repr__(self, include_packets_in_channel=True, include_arrived_packets=True):
        s = f"{self.channel_name}"
        if include_packets_in_channel:
            s += f"\n  Packets in channel: {self.packets_in_channel}"
        if include_arrived_packets:
            s += f"\n  Arrived packets: {self.arrived_packets}"
        return s

    def sim_print(self, message: str, time: int):
        return #! DEBUG
        print(f"[{time}] {self.channel_name}: {message}")

class ForwardChannel(Channel):
    epsilon: float # Probability of packet loss
    dropped_packets: list[RLNCPacket]

    def __init__(self, propagation_delay: int, epsilon: float, hop_index: int, path_index_in_hop: int, **kwargs):
        super().__init__(propagation_delay, hop_index, path_index_in_hop, **kwargs)
        self.epsilon = epsilon
        self.dropped_packets = []
        self.pending_packets_buffer = []
        self.channel_name = "Forward" + self.channel_name

        # for debug:
        self.num_packets_to_drop_per_window = 10 * epsilon
        self.t = 0
    
    def add_packets_to_pending_packets_buffer(self, packets: list[RLNCPacket], current_time: int):
        """ Add packets to pending_packets_buffer """
        self.pending_packets_buffer.extend(packets)
        self.sim_print(f"Packets added to pending_packets_buffer:\n\t{packets}", current_time)

    def apply_noise_on_single_packet(self, packet: RLNCPacket) -> tuple[RLNCPacket, bool]:
        dropped = False
        if random.random() < self.epsilon:
                self.dropped_packets.append(packet)
                dropped = True
        return packet, dropped

    # def apply_noise_on_single_packet(self, packet: RLNCPacket) -> tuple[RLNCPacket, bool]:
    #     """ For debug: drop packets in a window of 10 packets """
    #     self.t += 1
    #     place_in_dropping_window = self.t % 10
    #     dropped = False
    #     if place_in_dropping_window >= (10 - self.num_packets_to_drop_per_window):
    #         dropped = True
    #         self.dropped_packets.append(packet)
    #     return packet, dropped


    def run_step(self, current_time: int) -> RLNCPacket | None:
        """Run step for forward channel
        Returns: The packet that was sent on this step, or None if the packet was dropped by the forward channel"""
        # Propagate packets in channel
        super().run_step()
        # Add packet from pending_packets_buffer to channel so it'll start propagating from next step
        packet = None 
        if len(self.pending_packets_buffer) > 0:
            packet = self.pending_packets_buffer.pop(0)
            # Apply noise on packet
            packet, dropped = self.apply_noise_on_single_packet(packet)
            if not dropped:
                # Add new packet to channel if it is not dropped
                packet.set_creation_time(current_time) # Set creation time to actual transmission time
                super().add_packets_to_channel([packet], current_time)
            else:
                self.sim_print(f"Packet dropped:\n\t{packet}", current_time)
        return packet
        
    def get_dropped_packets(self):
        return self.dropped_packets

    def get_channel_history(self, include_dropped_packets: bool = False) -> list[RLNCPacket]:
        undropped_history = super().get_channel_history()
        if include_dropped_packets:
            return undropped_history + self.dropped_packets
        else:
            return undropped_history
    
    def __repr__(self, include_packets_in_channel=True, include_arrived_packets=True, include_dropped_packets=True):
        s = f"{self.channel_name}"
        s = super().__repr__(include_packets_in_channel, include_arrived_packets)
        s += f"\n  Epsilon: {self.epsilon}"
        if include_dropped_packets:
            s += f"\n  Dropped packets: {self.dropped_packets}"
        return s