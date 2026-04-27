from Packet import RLNCPacket, FeedbackPacket, RLNCType, NodeRLNCType, FeedbackType, PacketID
from Channels import Channel, ForwardChannel, Path
from CodedEquation import CodedEquation
from copy import deepcopy


class GeneralSenderPath(Path):
    def __init__(self, 
                path: Path, 
                my_sender: 'GeneralSender', 
                path_index: int,
                initial_epsilon: float = 0.0):
        self.__dict__.update(path.__dict__) # Automatically copy all attributes from the existing path
        self.my_sender = my_sender
        self.unit_name = f"{self.my_sender.unit_name}.SenderPath[{path_index}]"
        self.path_index = path_index # Index of the path in the sender side of the hop
        self.sent_channel_history = [] # History of sent packets
        self.received_feedback_history = [] # History of received feedback packets
        self.pending_packets_to_send : list[RLNCPacket] = [] # Packets that are pending for transmission
        self.all_feedback_history : list[FeedbackPacket] = [] # History of all feedback packets received from this path
        self.acked_feedback_history : list[FeedbackPacket] = [] # History of all ACK feedback packets received from this path
        self.nacked_feedback_history : list[FeedbackPacket] = [] # History of all NACK feedback packets received from this path
        self.current_feedbacks : list[FeedbackPacket] = [] # Feedback packets received in the current step
        self.epsilon_est : float = initial_epsilon
        # self.r : float = 1.0 - initial_epsilon # Forward Channel rate
        self.update_path_params(is_init=True)

    def update_path_params(self, is_init: bool = False):
        if len(self.all_feedback_history) > 0:
            self.epsilon_est = len(self.nacked_feedback_history) / len(self.all_feedback_history)
            self.r = 1.0 - self.epsilon_est
        elif is_init:
            self.r = 1.0 - self.epsilon_est
    
    def run_feedback_channel_step(self):
        """ Run feedback channel step, get feedbacks and update path params """
        self.feedback_channel.run_step()
        self.current_feedbacks = self.pop_arrived_feedback()
        if self.current_feedbacks is not None:
            self.all_feedback_history.extend(deepcopy(self.current_feedbacks))
            self.acked_feedback_history.extend([fb for fb in self.current_feedbacks if fb.type == FeedbackType.ACK])
            self.nacked_feedback_history.extend([fb for fb in self.current_feedbacks if fb.type == FeedbackType.NACK])
            self.update_path_params()
        else:
            self.sim_print(f"No feedbacks arrived on path {self.path_index}")

    def pop_arrived_feedback(self):
        feedbacks = self.feedback_channel.pop_arrived_packets()
        return feedbacks if feedbacks is not None else []
    
    def add_packet_to_sent_channel_history(self, packet: RLNCPacket):
        self.sent_channel_history.append(deepcopy(packet))

    def add_packet_to_received_feedback_history(self, packet: FeedbackPacket):
        self.received_feedback_history.append(deepcopy(packet))

    def get_sent_channel_history(self):
        return self.sent_channel_history

    def get_received_feedback_history(self):
        return self.received_feedback_history  

    def get_current_feedbacks(self):
        return self.current_feedbacks

    def get_params(self) -> dict[str, any]:
        return {
            "unit_name": self.unit_name,
            "num_pending_packets_to_send": len(self.pending_packets_to_send),
            "num_all_feedback_history": len(self.all_feedback_history),
            "num_acked_feedback_history": len(self.acked_feedback_history),
            "num_nacked_feedback_history": len(self.nacked_feedback_history),
            "num_current_feedbacks": len(self.current_feedbacks),
            "num_sent_channel_history": len(self.sent_channel_history),
            "num_received_feedback_history": len(self.received_feedback_history),
            "epsilon_est": self.epsilon_est,
            "r": self.r,
        }

    def sim_print(self, message: str, time: int=None):
        # print(f"[{self.my_sender.t}] {self.unit_name}: {message}")
        t = time if time is not None else self.my_sender.t
        super().sim_print(message, t)


class SimSenderPath(GeneralSenderPath):
    def __init__(self,
                path: Path,
                my_sender: 'SimSender',
                path_index: int,
                initial_epsilon: float = 0.0):
        super().__init__(path, my_sender, path_index)
        self.mp : int = 0
        self.unit_name = f"{self.my_sender.unit_name}.SenderPath[{path_index}]"

    # def run_feedback_channel_step(self):
    #     """ Run feedback channel step, get feedbacks and update path params """
    #     super().run_feedback_channel_step()
    #     if self.current_feedbacks is not None:
    #         self.update_path_params()

    def get_mp(self):
        return self.mp

    # def run_forward_channel_step(self, current_time: int) -> RLNCPacket:
    #     """Run forward channel step and updates new transmitions statistics"""
    #     rlnc_packet = super().run_forward_channel_step(current_time)
    #     if rlnc_packet is not None:
    #         self.my_sender.new_transmission_updates(rlnc_packet)
    #     return rlnc_packet  

    def get_params(self) -> dict[str, any]:
        params = super().get_params()
        params.update({
            "mp": self.mp
        })
        return params


class GeneralSender:
    def __init__(
        self,
        rtt: int,
        paths: list[Path],
        init_paths: bool = False, # Default is False for SimSender
        initial_epsilon: float = 0.0,
        ):
        self.t = 0

        # Constants
        self.rtt = rtt
        self.initial_epsilon = initial_epsilon
        if init_paths:
            self.paths : list[GeneralSenderPath] = [GeneralSenderPath(path, self, i, initial_epsilon) for i, path in enumerate(paths)]
        else:
            self.paths = paths
        
        # Feedback tracking
        self.feedbacks : list[FeedbackPacket] = [] # Feedback packets from all paths at current time
        
        # Transmission tracking
        self.latest_rlnc_packet_on_air = RLNCPacket(global_path_id=None, # For EW
                                            type=None,
                                            information_packets=[],
                                            prop_time_left_in_channel=None,
                                            creation_time=0)
        
        # Statistics
        self.all_feedback_history : list[FeedbackPacket] = []
        self.inforamtion_packets_first_transmission_times : dict[int, int] = {} # Time of first transmission of each information packet

    def run_step(self, time: int=None):
        """Update time, get feedbacks from all paths and update r for each path"""
        # Update time
        if time is not None:
            self.t = time
        else:
            self.t += 1
        # Get feedbacks from all fb channels and update r for each path
        self.get_feedbacks_from_all_paths()

    def get_feedbacks_from_all_paths(self):
        # Get feedbacks from the feedback channels
        self.feedbacks = [] # Clear feedbacks from previous step
        for path in self.paths:
            # Update r and get feedbak from path 
            path.run_feedback_channel_step()
            feedbacks = path.get_current_feedbacks()
            # Drop feedbacks on packets that wasn't sent (at the end of simulation)
            feedbacks = [fb for fb in feedbacks if fb.get_related_packet_id().get_creation_time() <= self.latest_rlnc_packet_on_air.get_creation_time()]
            self.feedbacks.extend(feedbacks) # Save current feedbacks
        self.all_feedback_history.extend(deepcopy(self.feedbacks))

    def send_packet(self, path: GeneralSenderPath, packet: RLNCPacket):
        # self.update_information_packets_first_transmission_times(packet)
        path.add_packet_to_forward_channel(packet, current_time=self.t)
        rlnc_packet = path.run_forward_channel_step(current_time=self.t)
        if rlnc_packet is not None:
            self.new_transmission_updates(rlnc_packet)
        self.latest_rlnc_packet_on_air = packet

    def new_transmission_updates(self, rlnc_packet_to_send: RLNCPacket):
        self.update_information_packets_first_transmission_times(rlnc_packet_to_send)
        # self.latest_rlnc_packet_on_air = rlnc_packet_to_send

    def update_information_packets_first_transmission_times(self, new_rlnc_packet: RLNCPacket):
        for packet in new_rlnc_packet.get_information_packets():
            if packet not in self.inforamtion_packets_first_transmission_times:
                self.inforamtion_packets_first_transmission_times[packet] = self.t

    def sim_print(self, message: str):
        print(f"[{self.t}] {self.unit_name}: {message}")

class SimSender(GeneralSender):
    def __init__(
        self,
        num_of_packets_to_send: int,
        rtt: int,
        paths: list[Path],
        initial_epsilon: float = 0.0,
        max_allowed_overlap: int = None,
        threshold: float = 0.0,
        network = None,
        next_hop : 'SimReceiver | Node' = None,
        ):
        super().__init__(rtt, paths, init_paths=False, initial_epsilon=initial_epsilon)
        # Sender constants
        self.unit_name = "SimSender"
        self.num_of_packets_to_send = num_of_packets_to_send
        self.paths = [SimSenderPath(path, self, i, initial_epsilon) for i, path in enumerate(paths)]
        self.num_of_paths = len(self.paths)
        self.my_network = network
        self.next_hop = next_hop

        # Receiver state tracking
        self.acked_equations: dict[PacketID, CodedEquation] = {} # Acked equations
        self.equations_waiting_feedback: dict[PacketID, CodedEquation] = {} # Equations that are waiting for feedback
        self.decoded_information_packets_history : list[int] = []
        
        # Sender Parameters tracking
        self.rlnc_ids_depended_on_undecoded_information_packets : list[PacketID] = [] # RLNC packets that are dependent on uncoded information packets
        self.sent_fec_history : list[RLNCPacket] = []
        self.sent_fb_fec_history : list[RLNCPacket] = []
        self.sent_new_rlnc_history : list[RLNCPacket] = [] # For determining ad1
        self.acked_feedback_history : list[FeedbackPacket] = []
        self.nacked_feedback_history : list[FeedbackPacket] = []
        
        # FB-FEC parameters
        self.remaining_paths_for_transmission = self.paths # List of paths that are free for transmission
        self.ad1 = 0.0 # Added DoF with feedback
        self.ad2 = 0.0 # Added DoF without feedback
        self.adg = self.ad1 + self.ad2 # (=0)
        self.md1 = 0.0 # Missing DoF with feedback
        self.md2 = 0.0 # Missing DoF without feedback
        self.mdg = self.md1 + self.md2 # (=0)
        self.d = 0.0
        self.threshold = threshold
        self.delta = self.num_of_paths * ( self.d - 1 - self.threshold)

        # FEC parameters
        self.EW = len(self.paths) * (rtt - 1) # End window of k=P*(RTT-1) new packets 
        self.max_overlap_flag = False # Flag to indicate if max overlap has been reached
        self.max_allowed_overlap = max_allowed_overlap if max_allowed_overlap is not None else 2 * rtt # Denoted as o_bar in the paper
        # self.feedbacks : list[FeedbackPacket] = [] # Feedback packets from all paths at current time
        self.oldest_information_packet_on_air = 1 # Last information packet sent- for max overlap
        self.newest_information_packet_on_air = 0 # Newest information packet sent- for max overlap
        self.num_rlnc_until_ew = 0
        
        # Statistics
        self.parameters_history = {
            "md1": [self.md1],
            "md2": [self.md2],
            "ad1": [self.ad1],
            "ad2": [self.ad2],
            "delta": [self.delta],
            "d": [self.d],
            "mdg": [self.mdg],
            "adg": [self.adg],
            "rlnc_id_depended_on_undecoded_information_packets": [self.rlnc_ids_depended_on_undecoded_information_packets],
        }
    
    
    def run_step(self, time: int=None):
        super().run_step(time)
        # On MP MH networks - Send global paths ranking to the network
        if hasattr(self.my_network, 'update_natural_matching'):
            self.update_natural_matching()
        # Figure out which packets arrived to receiver
        self.infer_receiver_state() 
        # Update parameters from received feedbacks
        self.update_sim_sender_params()
        self.remaining_paths_for_transmission = self.paths
        # Handle max overlap
        if self.is_max_overlap():
            self.handle_max_overlap()
        # No overlap, send packets until end of simulation
        elif len(self.decoded_information_packets_history) < self.num_of_packets_to_send:
            self.fec_transmissions()
            if len(self.remaining_paths_for_transmission) > 0:
                # FB-FEC transmission
                if self.delta > 0: # FEC transmission
                    self.fb_fec_transmissions()
                self.new_transmissions()
                # Init FEC transmissions
                self.init_fec_transmissions()
        
        # Run all units steps 
        self.run_remaining_paths_and_receiver_step()
        
    def update_natural_matching(self):
        paths_by_r = sorted(self.paths, key=lambda path: path.r, reverse=True)
        self.my_network.update_natural_matching([path.get_global_path_index() for path in paths_by_r])

    def is_max_overlap(self):
        # Check if max overlap has been reached un current time and raise flag
        if self.newest_information_packet_on_air - self.oldest_information_packet_on_air + 1 > self.max_allowed_overlap:
            self.max_overlap_flag = True
            return True 
        # max overlap reached in earlier time
        elif self.max_overlap_flag: 
            # Check if there are still DoF
            unknown_packets_to_decode = self.get_unknown_packets_to_decode()
            if self.is_decodable_set_of_equations(unknown_packets_to_decode):
                self.max_overlap_flag = False
                return True
            else:
                return False
    
    def handle_max_overlap(self):
        # Send same FEC RLNC on all channels
        for path in self.paths:
            self.create_and_send_rlnc(path, RLNCType.FEC)
        # Clear remaining paths since all paths sent
        self.remaining_paths_for_transmission = []
    
    def fec_transmissions(self):
        paths_for_fec_transmission = [path for path in self.remaining_paths_for_transmission if path.mp > 0]
        for path in paths_for_fec_transmission:
            self.create_and_send_rlnc(path, RLNCType.FEC)
            path.mp -= 1
        # Remove paths that have been used for FEC transmission
        self.remaining_paths_for_transmission = list(set(self.remaining_paths_for_transmission) - set(paths_for_fec_transmission))
    
    def fb_fec_transmissions(self):
        paths_for_fb_fec_transmission = self.perform_bit_filling()
        for path in paths_for_fb_fec_transmission:
            self.create_and_send_rlnc(path, RLNCType.FB_FEC)
        # Remove paths that have been used for FB-FEC transmission
        # self.remaining_paths_for_transmission = list(set(self.remaining_paths_for_transmission) - set(paths_for_fb_fec_transmission))
        self.remaining_paths_for_transmission = [
            p for p in self.remaining_paths_for_transmission 
            if p not in paths_for_fb_fec_transmission
        ]

    def perform_bit_filling(self) -> list[SimSenderPath]:
        self.remaining_paths_for_transmission.sort(key=lambda path: path.r)
        paths_for_fb_fec_transmission: list[SimSenderPath] = []
        cumulative_r = 0.0
        for path in self.remaining_paths_for_transmission:
            paths_for_fb_fec_transmission.append(path)
            cumulative_r += path.r
            if cumulative_r >= self.delta:
                break
        if cumulative_r < self.delta:
            self.sim_print(f"perform_bit_filling: Not enough paths to fill the delta. Cumulative r: {cumulative_r}, delta: {self.delta}")
        return paths_for_fb_fec_transmission

    def new_transmissions(self):
        paths_for_new_transmission = []
        # I'm going over the reversed list because after fb_fec_transmissions the list is sorted from lowest r to highest.
        # So in the end of the simulation, if there will not be enough packets to send, I want to send the packets from the paths with the highest r.
        for path in reversed(self.remaining_paths_for_transmission): 
        # for path in (self.remaining_paths_for_transmission): 
            if not self.is_EW():
                paths_for_new_transmission.append(path)
                self.create_and_send_rlnc(path, RLNCType.NEW)
            else: 
                break
        self.remaining_paths_for_transmission = list(set(self.remaining_paths_for_transmission) - set(paths_for_new_transmission))

    def is_EW(self):
        return self.num_rlnc_until_ew >= self.EW
    
    def init_fec_transmissions(self):
        if self.is_EW():
            # set mp for all paths after k = num_of_paths * (rtt - 1) transmissions
            for path in self.paths:
                path.mp = round(path.epsilon_est * (self.rtt - 1)) # Round to nearest integer
            # start FEC for all remaining paths
            paths_for_init_fec = list(self.remaining_paths_for_transmission)  # Copy the list
            for path in paths_for_init_fec:
                self.create_and_send_rlnc(path, RLNCType.FEC)
                path.mp -= 1 # Decrease mp by 1 because the packet has been transmitted
            # Remove paths that sent FEC from remaining_paths_for_transmission
            self.remaining_paths_for_transmission = list(set(self.remaining_paths_for_transmission) - set(paths_for_init_fec))
            self.num_rlnc_until_ew = 0

    def infer_receiver_state(self):
        """ Infer receiver state from feedbacks """
        # Delete nacked equations
        self.sim_print(f"Infer_receiver_state: equations_waiting_feedback before deleting nacked equations: {self.equations_waiting_feedback}")
        nack_feedbacks = [nack for nack in self.feedbacks if nack.type == FeedbackType.NACK]
        self.sim_print(f"Infer_receiver_state: nack_feedbacks: {nack_feedbacks}")
        for nack in nack_feedbacks:
            related_equation = nack.get_related_packet_id()
            # Only pop if equation exists (may have been cleaned up already)
            if related_equation in self.equations_waiting_feedback:
                self.equations_waiting_feedback.pop(related_equation)
            else:
                self.sim_print(f"Infer_receiver_state: NACK for already removed equation: {related_equation}")
        self.sim_print(f"Infer_receiver_state: equations_waiting_feedback after deleting nacked equations: {self.equations_waiting_feedback}")

        # Collect ACKs from feedbacks
        acked_feedbacks = [pkt for pkt in self.feedbacks if pkt.is_ack()]
        self.sim_print(f"Infer_receiver_state: acked_feedbacks before trimming: {acked_feedbacks}")
        self.sim_print(f"Infer_receiver_state trimming by oldest_information_packet_on_air: {self.oldest_information_packet_on_air}")
        
        # TODO: No need for this part because afterwards we are moving the equation from equations_waiting_feedback, which are not trimmed
        # Trim all acks for already decoded equations
        trimmed_acked_feedbacks = []
        for ack in acked_feedbacks:
            # Keep only information packets newer than the most recent decoded packet
            ack.related_information_packets = [
                pkt for pkt in ack.get_related_information_packets()
                if pkt >= self.oldest_information_packet_on_air
            ]
            if len(ack.get_related_information_packets()) > 0:
                trimmed_acked_feedbacks.append(ack)
        acked_feedbacks = trimmed_acked_feedbacks
        self.sim_print(f"Infer_receiver_state: acked_feedbacks after trimming: {acked_feedbacks}")
        
        # Move acked equations to acked_equations
        for ack in acked_feedbacks:
            self.sim_print(f"Infer_receiver_state: ACK detected: {ack}")
            related_equation = ack.get_related_packet_id()
            self.acked_equations[related_equation] = self.equations_waiting_feedback.pop(related_equation)
        self.sim_print(f"Infer_receiver_state: equations_waiting_feedback after adding acked equations: {self.equations_waiting_feedback}")
        
        # Infer which equations can be decoded and decode them
        self.sim_print(f"Infer_receiver_state: acked_equations before inferring: {self.acked_equations}")
        # Collect all packets for decode 
        unknown_packets_to_decode = self.get_unknown_packets_to_decode()
        # Check if the receiver can decode all equations
        if self.is_decodable_set_of_equations(unknown_packets_to_decode): # Found a decodable set of equations: #unknowns <= #equations
            self.sim_print(f"Infer_receiver_state: Receiver can decode all equations.\n\
                len(unknown_packets_to_decode): {len(unknown_packets_to_decode)}, len(self.acked_equations): {len(self.acked_equations)}")
            
            # All acked equations can be decoded
            self.acked_equations.clear()
                
            # Update Sender
            latest_decoded_inforamtion_packet = max(unknown_packets_to_decode)
            self.oldest_information_packet_on_air = latest_decoded_inforamtion_packet + 1
            self.decoded_information_packets_history.extend(list(unknown_packets_to_decode))
            self.decoded_information_packets_history.sort()
            
            # Remove decoded packets from all equations that they appear in
            for eq in list(self.equations_waiting_feedback.values()):
                eq.unknown_packets = [pkt for pkt in eq.unknown_packets if pkt > latest_decoded_inforamtion_packet]
                if len(eq.unknown_packets) == 0:
                    self.equations_waiting_feedback.pop(eq.get_related_rlnc_packet_id())
                    self.sim_print(
                        f"Infer_receiver_state: Removed Equation {eq.get_related_rlnc_packet_id()} "
                        f"from equations_waiting_feedback because it has no unknown packets\n"
                        f"  equation: {eq}"
                    )

    def get_unknown_packets_to_decode(self) -> set[int]:
        """ Returns a set of coded information packets that are not decoded yet.
        This set is made from all acked equations minus packets that are already decoded. """
        unknown_packets_to_decode = set() # Set is for eliminating duplicates
        for eq in self.acked_equations.values():
            # Add only information packets that are not decoded yet
            unknown_packets_to_decode.update(set(eq.get_unknown_packets()) - set(self.decoded_information_packets_history))
        return unknown_packets_to_decode

    def is_decodable_set_of_equations(self, unknown_packets_to_decode: set[int]):
        """ Checks if the set of unknown packets to decode is decodable.
        This is done by checking if the number of unknown packets is less than or equal to the number of acked equations. """
        if len(unknown_packets_to_decode) == 0:
            return False
        if len(unknown_packets_to_decode) <= len(self.acked_equations):
            return True
        return False
    
    def get_feedbacks_from_all_paths(self):
        super().get_feedbacks_from_all_paths()
        acks = [ack for ack in self.feedbacks if ack.is_ack()]
        nacks = [nack for nack in self.feedbacks if nack.is_nack()]
        self.acked_feedback_history.extend(deepcopy(acks))
        self.nacked_feedback_history.extend(deepcopy(nacks))

    def update_sim_sender_params(self):
        self.update_rlnc_id_depended_on_undecoded_information_packets()
        self.update_md1()
        self.update_md2()
        self.update_ad1()
        self.update_ad2()
        self.update_delta()

    def update_rlnc_id_depended_on_undecoded_information_packets(self):
        # Collect all RLNC history
        all_rlnc_history = self.get_all_rlnc_history()

        # Iterate over RLNC history and find which packets have undecoded information packets
        tmp_rlnc_ids_depended_on_undecoded_information_packets: set[PacketID] = set()
        sender_decoded_information_packets_set = set(self.decoded_information_packets_history)
        for rlnc_pkt in all_rlnc_history:
            info_pkts = set(rlnc_pkt.get_information_packets())
            # Check if RLNC got any undecoded information packets
            if not info_pkts.issubset(sender_decoded_information_packets_set):
                # Got undecoded information packets, save it's PacketID
                tmp_rlnc_ids_depended_on_undecoded_information_packets.add(rlnc_pkt.get_id())
        
        # Update rlnc_ids_depended_on_undecoded_information_packets
        self.rlnc_ids_depended_on_undecoded_information_packets = list(tmp_rlnc_ids_depended_on_undecoded_information_packets)
        self.parameters_history["rlnc_id_depended_on_undecoded_information_packets"].append(self.rlnc_ids_depended_on_undecoded_information_packets)
        
        # # Get all depended RLNC IDs
        # acked_equations = list(self.acked_equations.values())
        # equations_waiting_feedback = list(self.equations_waiting_feedback.values())
        # # Update the list of RLNC IDs that are depended on uncoded information packets
        # self.rlnc_ids_depended_on_undecoded_information_packets = \
        #     [eq.related_rlnc_packet_id for eq in acked_equations] + \
        #     [eq.related_rlnc_packet_id for eq in equations_waiting_feedback]
        # self.parameters_history["rlnc_id_depended_on_undecoded_information_packets"].append(self.rlnc_ids_depended_on_undecoded_information_packets)
    
    def update_md1(self):
        # Get all IDs for calculating md1
        nacked_rlnc_ids = [pkt.get_related_packet_id() for pkt in self.nacked_feedback_history]
        new_rlnc_ids = self.get_new_rlnc_ids()
        # md1 = |N & C^n & U|
        self.md1 = len(set(nacked_rlnc_ids) & \
            set(new_rlnc_ids) & \
            set(self.rlnc_ids_depended_on_undecoded_information_packets) \
        )
        self.parameters_history["md1"].append(self.md1)

    def update_md2(self):
        # Get all sets for calculating md2
        new_rlnc_ids = self.get_new_rlnc_ids()
        equations_waiting_feedback = list(self.equations_waiting_feedback.values())
        rlnc_ids_waiting_feedback = [eq.related_rlnc_packet_id for eq in equations_waiting_feedback]
        # Calc inter term for each path and sum them up
        md2_temp = 0
        for path in self.paths:
            rlnc_pkts_ids_from_path = [pkt.get_id() for pkt in path.get_forward_channel_history(include_dropped_packets=True)]
            # intersection = P_p & C^n & F & U
            intersection = set(rlnc_pkts_ids_from_path) & \
                            set(new_rlnc_ids) & \
                            set(rlnc_ids_waiting_feedback) & \
                            set(self.rlnc_ids_depended_on_undecoded_information_packets)
            md2_temp += path.epsilon_est * len(intersection)
        self.md2 = md2_temp
        self.parameters_history["md2"].append(self.md2)

    def update_ad1(self):
        acked_rlnc_ids = [ack.get_related_packet_id() for ack in self.acked_feedback_history]
        repeated_rlnc_ids = self.get_repeated_rlnc_ids()
        # ad1 = |A & C^r & U|
        self.ad1 = len(set(acked_rlnc_ids) & set(repeated_rlnc_ids) & set(self.rlnc_ids_depended_on_undecoded_information_packets))
        self.parameters_history["ad1"].append(self.ad1)

    def update_ad2(self):
        # Get all sets for calculating ad2
        repeated_rlnc_ids = self.get_repeated_rlnc_ids()
        equations_waiting_feedback = list(self.equations_waiting_feedback.values())
        rlnc_ids_waiting_feedback = [eq.related_rlnc_packet_id for eq in equations_waiting_feedback]
        # Calc inter term for each path and sum them up
        ad2_temp = 0
        for path in self.paths:
            rlnc_pkts_from_path = [pkt.get_id() for pkt in path.get_forward_channel_history(include_dropped_packets=True)]
            # intersection = P_p & C^r & F & U
            intersection = set(rlnc_pkts_from_path) & \
                            set(repeated_rlnc_ids) & \
                            set(rlnc_ids_waiting_feedback) & \
                            set(self.rlnc_ids_depended_on_undecoded_information_packets)
            ad2_temp += path.r * len(intersection)
        self.ad2 = ad2_temp
        self.parameters_history["ad2"].append(self.ad2)

    def update_delta(self):
        self.mdg = self.md1 + self.md2
        self.adg = self.ad1 + self.ad2
        try:
            self.d = self.mdg / self.adg
        except ZeroDivisionError:
            self.d = self.d
        self.delta = self.num_of_paths * ( self.d - 1 - self.threshold)
        self.parameters_history["delta"].append(self.delta)
        self.parameters_history["d"].append(self.d)
        self.parameters_history["mdg"].append(self.mdg)
        self.parameters_history["adg"].append(self.adg)

    def get_new_rlnc_ids(self):
        return [pkt.get_id() for pkt in self.sent_new_rlnc_history]

    def get_repeated_rlnc_ids(self):
        return [pkt.get_id() for pkt in (self.sent_fec_history + self.sent_fb_fec_history)]

    def eliminate_seen_packets(self, packets: list[FeedbackPacket]):
        """ Go over each path's pending packets and eliminate the packets that have been acknowledged """
        raise NotImplementedError("Not implemented")

    def create_and_send_rlnc(self, path: SimSenderPath, type: RLNCType):
        if type == RLNCType.NEW:
            end_of_sim_adjustment = 1 if self.newest_information_packet_on_air == self.num_of_packets_to_send else 2
            information_packets = list(range(self.oldest_information_packet_on_air, self.newest_information_packet_on_air + end_of_sim_adjustment))
        else: # FEC or FB-FEC
            information_packets = self.latest_rlnc_packet_on_air.get_information_packets()

        rlnc_packet_to_send = RLNCPacket(
            global_path_id=path.get_global_path_index(),
            type=type,
            information_packets=information_packets,
            prop_time_left_in_channel=path.get_propagation_delay(),
            creation_time=self.t
        )
        self.send_packet(path, rlnc_packet_to_send)

    def add_equation_to_waiting_feedback(self, rlnc_packet: RLNCPacket):
        equation = CodedEquation(rlnc_packet.get_id(), rlnc_packet.get_information_packets())
        assert self.equations_waiting_feedback.get(rlnc_packet.get_id()) is None, \
            f"Equation {rlnc_packet.get_id()} already exists in equations_waiting_feedback, equation: {self.equations_waiting_feedback[rlnc_packet.get_id()]}"
        self.equations_waiting_feedback[rlnc_packet.get_id()] = equation

    def add_rlnc_packet_to_history(self, rlnc_packet: RLNCPacket):
        match rlnc_packet.get_type():
            case RLNCType.NEW:
                self.sent_new_rlnc_history.append(rlnc_packet)
            case RLNCType.FEC:
                self.sent_fec_history.append(rlnc_packet)
            case RLNCType.FB_FEC:
                self.sent_fb_fec_history.append(rlnc_packet)
            case _:  # Defensive: catch any invalid types
                raise ValueError(f"Invalid RLNC packet type: {rlnc_packet.get_type()}, packet:\n    {rlnc_packet}")

    def new_transmission_updates(self, rlnc_packet_to_send: RLNCPacket):
        # self.latest_rlnc_packet_on_air = rlnc_packet_to_send
        super().new_transmission_updates(rlnc_packet_to_send)
        if rlnc_packet_to_send.get_type() == RLNCType.NEW:
            self.num_rlnc_until_ew += 1
        # self.update_information_packets_first_transmission_times(rlnc_packet_to_send)
        self.add_equation_to_waiting_feedback(rlnc_packet_to_send)
        self.add_rlnc_packet_to_history(rlnc_packet_to_send)
        self.update_latest_information_packet_on_air(rlnc_packet_to_send)
    
    # def update_information_packets_first_transmission_times(self, new_rlnc_packet: RLNCPacket):
    #     for packet in new_rlnc_packet.get_information_packets():
    #         if packet not in self.inforamtion_packets_first_transmission_times:
    #             self.inforamtion_packets_first_transmission_times[packet] = self.t
    
    def update_latest_information_packet_on_air(self, new_rlnc_packet: RLNCPacket):
        latest_information_packet_in_new_rlnc_packet = max(new_rlnc_packet.get_information_packets())
        self.newest_information_packet_on_air = max(self.newest_information_packet_on_air, latest_information_packet_in_new_rlnc_packet)

    def run_remaining_paths_and_receiver_step(self):
        # Run step for all remaining paths and receiver
        for path in self.remaining_paths_for_transmission: # When the simulation ends, there will be no remaining paths for transmission
            path.run_forward_channel_step(current_time=self.t)
        self.next_hop.run_step()

    def get_all_rlnc_history(self) -> list[RLNCPacket]:
        return self.sent_new_rlnc_history + self.sent_fec_history + self.sent_fb_fec_history

    # def sim_print(self, message: str):
    #     pass

    def __repr__(self):
        s = "SimSender:"
        s += f"\n  num_of_packets_to_send: {self.num_of_packets_to_send}"
        s += f"\n  rtt: {self.rtt}"
        s += f"\n  num paths: {self.num_of_paths}"
        s += f"\n  initial epsilon: {self.initial_epsilon}"
        s += f"\n  my receiver: {self.unit_name}"
        # num_of_trasmissions = len(self.sent_new_rlnc_history + self.sent_fec_history + self.sent_fb_fec_history)
        if self.t > 0:
            total_information_delivered = len(self.decoded_information_packets_history)
            s += f"\n  normalized throughput for t{self.t}: {total_information_delivered / self.t}"
        return s


class NodeSender(GeneralSender):
    def __init__(
        self,
        rtt: int,
        hop_num: int,
        paths: list[Path],
        initial_epsilon: float = 0.0,
        unit_name: str=None,
        parent_node: 'Node'=None,
        ):
        # Constants
        if unit_name is None: # Set unit name before calling super() for setting name that is not "GeneralReceiver"
            unit_name = f"NodeSender[{hop_num}]"
        self.unit_name = unit_name
        super().__init__(rtt, paths, init_paths=True, initial_epsilon=initial_epsilon)
        self.hop_num = hop_num

        # Network
        self.parent_node = parent_node

        # Buffers
        self.new_information_packets_buffer : set[int] = set()
        self.correction_information_packets_buffer : set[int] = set()

        # Statistics
        self.new_rlnc_packets_history : list[RLNCPacket] = []
        self.new_information_packets_history : set[int] = set()
        self.correction_packets_history : list[RLNCPacket] = []
        self.correction_information_packets_history : set[int] = set()

    def run_step(self, time: int=None):
        super().run_step(time)  # Update time, get feedbacks, and update r for each path
        self.infer_receiver_state() # Not implemented for 1st step
        self.update_buffers_and_history()
        self.perform_natural_matching()
        self.create_and_send_rlnc_on_all_paths()

    def create_and_send_rlnc_on_all_paths(self):
        global_path_pkt_rlnc_types = self.get_global_paths_rlnc_types()
        if not global_path_pkt_rlnc_types: # Packets were not received yet
            return

        for path in self.paths:
            global_path_idx = path.get_global_path_index()
            global_path_rlnc_type = global_path_pkt_rlnc_types.get(global_path_idx, None)
            assert global_path_rlnc_type is not None, \
                f"Global path {global_path_idx} has no RLNC type, global_path_pkt_rlnc_types: {global_path_pkt_rlnc_types}"

            rlnc_packet_to_send = self.create_rlnc(path, global_path_rlnc_type)

            if rlnc_packet_to_send is not None:
                self.send_packet(path, rlnc_packet_to_send)
                self.add_rlnc_packet_to_history(rlnc_packet_to_send)
            else:
                path.run_forward_channel_step(current_time=self.t)
        
    def add_rlnc_packet_to_history(self, rlnc_packet: RLNCPacket):
        if rlnc_packet.get_type() == RLNCType.NEW:
            self.new_rlnc_packets_history.append(rlnc_packet)
        else:
            self.correction_packets_history.append(rlnc_packet)
    
    def create_rlnc(self, path: GeneralSenderPath, rlnc_type: RLNCType) -> RLNCPacket | None:
        information_packets: list[int] = []
        rlnc_type_to_send: RLNCType = None
        if rlnc_type == RLNCType.NEW:
            information_packets = list(self.new_information_packets_buffer)
            rlnc_type_to_send = RLNCType.NEW
        else:
            rlnc_type_to_send = NodeRLNCType.CORRECTION
            information_packets = list(self.correction_information_packets_buffer)
        # Create RLNC only if packets had arrived
        if len(information_packets) > 0:
            rlnc_packet_to_send = RLNCPacket(
                global_path_id=path.get_global_path_index(),
                type=rlnc_type_to_send,
                information_packets=information_packets,
                prop_time_left_in_channel=path.get_propagation_delay(),
                creation_time=self.t
            )
            return rlnc_packet_to_send
        return None

    def update_buffers_and_history(self):
        """
        In step 1: Duplicate buffers from the NodeReceiver
        In step 2: Remove decoded information packets from buffers
        """
        new_information_packets = self.get_receiver_new_information_packets()
        correction_information_packets = self.get_receiver_correction_information_packets()
        
        # Update history
        self.new_information_packets_history.update(new_information_packets)
        self.correction_information_packets_history.update(correction_information_packets)

        # Update buffers
        # TODO: maybe if one buffer is empty, we should put the 2nd buffer's packets into the 1st buffer (at the start of the sim)
        self.new_information_packets_buffer = new_information_packets
        self.correction_information_packets_buffer = correction_information_packets

    def get_receiver_new_information_packets(self) -> set[int]:
        return self.parent_node.get_receiver_new_information_packets()

    def get_receiver_correction_information_packets(self) -> set[int]:
        return self.parent_node.get_receiver_correction_information_packets()

    def perform_natural_matching(self):
        global_paths_idx_by_r = self.get_global_paths_by_r()
        
        # Sort local paths by r in descending order
        my_paths_by_r = sorted(self.paths, key=lambda path: path.r, reverse=True)
        assert len(global_paths_idx_by_r) == len(my_paths_by_r), \
            f"Global paths by r and my paths by r must have the same length, global_paths_idx_by_r: {global_paths_idx_by_r}, my_paths_by_r: {my_paths_by_r}"
        
        # Match local paths to global paths by r
        for my_path, global_path_idx in zip(my_paths_by_r, global_paths_idx_by_r):
            my_path.set_global_path_index(global_path_idx)

    def get_global_paths_by_r(self) -> list[int]:
        return self.parent_node.get_global_paths_by_r()

    def get_global_paths_rlnc_types(self) -> dict[int, RLNCType]:
        return self.parent_node.get_global_paths_rlnc_types()

    def infer_receiver_state(self):
        pass