from Packet import RLNCPacket, FeedbackPacket, RLNCType, FeedbackType, PacketID
from Channels import Channel, ForwardChannel, Path
from CodedEquation import CodedEquation
from Sender import SimSender
from Receiver import GeneralReceiver, SimReceiver

from contextlib import redirect_stdout
import os
import sys
from dataclasses import dataclass

@dataclass
class SimulationStats:
    normalized_throughput : float = 0.0 # Normalized throughput at the end of simulation
    inorder_delay_mean : float = 0.0 # Mean inorder delay at the end of simulation
    inorder_delay_max : int = 0 # Max inorder delay at the end of simulation

    num_new_rlnc_packets : int = 0 # Number of new RLNC packets sent by sender
    num_fec_packets : int = 0 # Number of FEC packets sent by sender
    num_fb_fec_packets : int = 0 # Number of FB-FEC packets sent by sender
    num_transmissions : int = 0 # Number of transmissions at the end of simulation
    num_information_packets_sent : int = 0 # Total number of information packets sent by sender
    num_information_packets_decoded : int = 0 # Total number of information packets decoded by receiver
    num_transmissions_dropped : int = 0 # Number of transmissions that were dropped

class MPNetwork:
    def __init__(
        self,
        path_epsilons: list[float],
        initial_epsilon: float = None,
        max_iterations: int = None,
        num_packets_to_send: int = None,
        num_paths: int = 4,
        prop_delay: int = 2,
        threshold: float = 0.0,
        max_allowed_overlap: int = None,
    ):
        self.t = 0
        self.path_epsilons = path_epsilons
        self.num_paths = num_paths
        self.prop_delay = prop_delay
        self.rtt = prop_delay * 2
        self.threshold = threshold
        self.max_iterations = max_iterations
        if max_allowed_overlap is not None:
            assert max_allowed_overlap > 0
            self.max_allowed_overlap = max_allowed_overlap
        else:
            self.max_allowed_overlap = 2 * num_paths * (self.rtt - 1) # 2*k = 2*P*(RTT-1)
        
        if num_packets_to_send is not None:
            assert num_packets_to_send > 0
            self.num_packets_to_send = num_packets_to_send
        else:
            self.num_packets_to_send = sys.maxsize

        self.paths = [Path(prop_delay, epsilon, 0, i) for i, epsilon in enumerate(path_epsilons)]
        for i, path in enumerate(self.paths):
            path.set_global_path_index(i)
        
        self.receiver = SimReceiver(self.paths, self.rtt, unit_name="SimReceiver")
        init_eps = initial_epsilon if initial_epsilon is not None else 0.0
        self.sender = SimSender(self.num_packets_to_send, self.rtt, self.paths, max_allowed_overlap, threshold, 
                               receiver=self.receiver, initial_epsilon=init_eps)

        # Statistics
        self.sender_information_packets_sending_times : dict[int, int] = {} # Mapping for each information packet to the time it was sent
        self.sender_num_total_transmissions : int = 0 # Number of transmissions at the end of simulation
        self.sender_num_information_packets_sent : int = 0 # Total number of information packets sent by sender
        self.receiver_information_packets_decoding_times : dict[int, int] = {} # Mapping for each information packet to the time it was decoded
        self.receiver_num_information_packets_decoded : int = 0 # Total number of information packets delivered by receiver
        self.receiver_information_packets_by_decoding_time : dict[int, list[int]] = {} # Mapping for each decoding time to the list of information packets decoded
        self.normalized_throughput : float = 0.0 # Normalized throughput at the end of simulation
        self.inorder_delay_for_each_information_packet : dict[int, int] = {} # Mapping for each information packet to the inorder delay
        self.inorder_delay_mean : float = 0.0 # Mean inorder delay at the end of simulation
        self.inorder_delay_max : int = 0 # Max inorder delay at the end of simulation
        self.simulation_stats : SimulationStats = None # Simulation stats at the end of simulation
        self.sender_num_transmissions_dropped : int = 0 # Number of transmissions that were dropped

    def run_sim(self):
        if self.max_iterations is not None:
            for t in range(1, self.max_iterations + 1):
                with redirect_stdout(open(os.devnull, 'w')):
                    self.sender.run_step()
                if len(self.receiver.information_packets_decoding_times) >= self.num_packets_to_send:
                    break
            self.t = t
            self.collect_stats()
            print(f"Simulation completed at t={t} - all packets decoded")
        else:
            while len(self.receiver.information_packets_decoding_times) < self.num_packets_to_send:
                with redirect_stdout(open(os.devnull, 'w')):
                    self.t += 1
                    self.sender.run_step()
            self.collect_stats()
            print(f"Simulation completed at t={self.max_iterations} - all packets decoded")

    def collect_stats(self):
        self.collect_sender_stats()
        self.collect_receiver_stats()
        self.calculate_all_simulation_stats()
        self.simulation_stats = SimulationStats(
            normalized_throughput=self.normalized_throughput,
            inorder_delay_mean=self.inorder_delay_mean,
            inorder_delay_max=self.inorder_delay_max,
            num_new_rlnc_packets=len(self.sender.sent_new_rlnc_history),
            num_fec_packets=len(self.sender.sent_fec_history),
            num_fb_fec_packets=len(self.sender.sent_fb_fec_history),
            num_transmissions=self.sender_num_total_transmissions,
            num_information_packets_sent=self.sender_num_information_packets_sent,
            num_information_packets_decoded=self.receiver_num_information_packets_decoded,
            num_transmissions_dropped=self.sender_num_transmissions_dropped
        )

    def collect_sender_stats(self):
        # Get information packets sent by sender
        self.sender_information_packets_sending_times = self.sender.inforamtion_packets_first_transmission_times.copy()
        information_packets_sent_by_sender = sorted(list(self.sender_information_packets_sending_times.keys()))
        assert len(information_packets_sent_by_sender) == len(set(information_packets_sent_by_sender)), \
            f"Sender sent duplicate information packets:\n\t{information_packets_sent_by_sender}"
        # Save number of information packets sent by sender
        self.sender_num_information_packets_sent = len(information_packets_sent_by_sender)
        # Get total number of transmissions by sender
        all_trasmissions_by_sender = self.sender.sent_new_rlnc_history + self.sender.sent_fec_history + self.sender.sent_fb_fec_history
        self.sender_num_total_transmissions = len(all_trasmissions_by_sender)
        # Get last transmission time to arrive to receiver
        last_transmission_time_to_arrive_to_receiver = self.t - self.prop_delay
        # Get number of transmissions that arrived to receiver
        self.sender_num_transmissions_arrived_to_receiver = len([transmission for transmission in all_trasmissions_by_sender if transmission.get_creation_time() <= last_transmission_time_to_arrive_to_receiver])

        # Get number of transmissions that were dropped
        self.sender_num_transmissions_dropped = sum([len(path.get_dropped_packets()) for path in self.sender.paths])

    def collect_receiver_stats(self):
        self.receiver_information_packets_decoding_times = self.receiver.information_packets_decoding_times.copy()
        information_packets_decoded_by_receiver = sorted(list(self.receiver_information_packets_decoding_times.keys()))
        assert len(information_packets_decoded_by_receiver) == len(set(information_packets_decoded_by_receiver)), \
            f"Receiver decoded duplicate information packets:\n\t{information_packets_decoded_by_receiver}"
        self.receiver_num_information_packets_decoded = len(information_packets_decoded_by_receiver)

    def calculate_all_simulation_stats(self):
        self.calculate_normalized_throughput_stats()
        self.calculate_inorder_delays_stats()

    def calculate_normalized_throughput_stats(self):
        # Maybe there's no need for this:
        # Gather all decoded information packets at receiver by decoding time
        for packet, time in self.receiver_information_packets_decoding_times.items():
            self.receiver_information_packets_by_decoding_time.setdefault(time, []).append(packet)

        # Calculate normalized throughput
        # self.normalized_throughput = self.receiver_num_information_packets_decoded / self.sender_num_total_transmissions
        self.normalized_throughput = self.receiver_num_information_packets_decoded / self.t

    def calculate_inorder_delays_stats(self):
        # Calculate inorder delays for each information packet
        for packet, decode_time in self.receiver_information_packets_decoding_times.items():
            packet_sending_time = self.sender_information_packets_sending_times[packet]
            inorder_delay = decode_time - packet_sending_time
            self.inorder_delay_for_each_information_packet[packet] = inorder_delay
        # Calculate mean inorder delay
        self.inorder_delay_mean = sum(self.inorder_delay_for_each_information_packet.values()) / len(self.inorder_delay_for_each_information_packet)
        # Calculate max inorder delay
        self.inorder_delay_max = max(self.inorder_delay_for_each_information_packet.values())

    def get_simulation_stats(self) -> SimulationStats:
        """Get simulation statistics after run_sim() completes."""
        return self.simulation_stats