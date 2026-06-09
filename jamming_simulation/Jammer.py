import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from general.GeneralUnit import GeneralUnit
from JamChannels import JamPath

class Jammer(GeneralUnit):
    def __init__(
        self,
        paths: list[JamPath],
        rtt: int,
        alpha: int, # Blocking round time parameter. block time = RTT / alpha
        k: int, # Number of channels to block in each jamming round
        unit_name: str = "Jammer",
        t: int = 0,
        debug: bool = False,
        parent_network: 'Network' = None
        ):
        super().__init__(unit_name, t, debug)
        self.rtt = rtt
        self.k = k # Number of paths to jam in each jamming round
        self.alpha = alpha # Jamming round time parameter. jamming round time = RTT / alpha
        self.jamming_round_time = rtt / alpha
        self.parent_network = parent_network
        self.paths = paths
        self.jammed_paths: list[JamPath] = []
        self.jammed_paths_history: dict[int, list[JamPath]] = {}

        assert len(self.paths) >= self.k, f"Number of paths must be greater than or equal to the number of paths to jam. len(paths) = {len(self.paths)}, k = {self.k}"
        self.select_new_paths_to_jam()

    def run_step(self, time: int = None):
        super().run_step(time)
        # Choose new random channels to block every block time
        if self.t % self.jamming_round_time == 0:
            self.select_new_paths_to_jam()
        # Jam paths
        for path in self.jammed_paths:
            path.jam_path()

    def select_new_paths_to_jam(self):
        for path in self.jammed_paths:
            path.unjam_path()
        self.jammed_paths_history[self.t] = self.jammed_paths
        self.jammed_paths = random.sample(self.paths, self.k)

    def get_jammed_paths_history(self) -> dict[int, list[JamPath]]:
        return self.jammed_paths_history