from Channels import Path
from Receiver import NodeReceiver


class Node:
    def __init__(self,
                 hop_num: int,
                 input_paths: list[Path],
                 output_paths: list[Path],
                 rtt: int,
                 unit_name: str=None):
        self.hop_num = hop_num
        self.rtt = rtt
        self.unit_name = unit_name
        self.t = 0
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.receiver = NodeReceiver(hop_num, input_paths, rtt, unit_name, self)
        self.sender = NodeSender()

    def run_step(self, time: int=None):
        self.receiver.run_step(time)
        self.sender.run_step(time)
        
        