from Packet import RLNCType, NodeRLNCType
from Channels import Path
from Receiver import NodeReceiver
from Sender import NodeSender


class Node:
    def __init__(self,
                 hop_num: int,
                 input_paths: list[Path],
                 output_paths: list[Path],
                 rtt: int,
                 unit_name: str=None,
                 next_hop: 'Node | SimReceiver'=None,
                 Network: 'MpMhNetwork'=None):
        self.t = 0

        # Constants
        self.hop_num = hop_num
        self.unit_name = unit_name if unit_name is not None else f"Node[{hop_num}]"
        self.rtt = rtt
        self.input_paths = input_paths
        self.output_paths = output_paths

        # Network
        self.parent_network = Network
        self.next_hop = next_hop

        # Node units
        self.my_receiver = NodeReceiver(hop_num=hop_num, input_paths=input_paths, rtt=rtt, unit_name=unit_name, parent_node=self)
        self.my_sender = NodeSender(rtt=rtt, hop_num=hop_num, paths=output_paths, unit_name=unit_name, parent_node=self)

        # Natural Matching Tracking
        self.global_paths_rlnc_types : dict[int, NodeRLNCType | RLNCType] = {} # Mapping for each global path index to the RLNC type received on that path

    def run_step(self, time: int=None):
        self.my_receiver.run_step(time)
        self.global_paths_rlnc_types = self.my_receiver.get_global_paths_rlnc_types()
        self.my_sender.run_step(time)
        if hasattr(self.next_hop, 'run_step'):
            self.next_hop.run_step(time)

    def get_global_paths_by_r(self) -> list[int]:
        return self.parent_network.global_paths_idx_by_r

    def get_global_paths_rlnc_types(self) -> dict[int, NodeRLNCType | RLNCType]:
        return self.global_paths_rlnc_types

    def get_receiver_new_information_packets(self) -> set[int]:
        return self.my_receiver.new_information_packets_buffer

    def get_receiver_correction_information_packets(self) -> set[int]:
        return self.my_receiver.correction_information_packets_buffer