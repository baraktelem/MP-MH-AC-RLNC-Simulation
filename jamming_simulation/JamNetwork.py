"""JamMpMhNetwork: a multi-hop multipath network where intermediate nodes are
arranged as P independent chains and a Jammer can block any forward channel
before each tick.

Topology:
    paths[P][H]  - chain-major; paths[c][h] is chain c's JamPath at hop h
    nodes[P][H-1] - flat layout of plain Node objects, each with one input
                    path and one output path. next_hop=None on every Node so
                    Node.run_step does NOT cascade. JamMpMhNetwork.run_sim
                    orchestrates the time-step explicitly:

        1. jammer.run_step(time=t)
        2. network.sender.run_step()
        3. for h in 0..H-2:
               for c in 0..P-1:
                   nodes[c][h].run_step(time=t)
        4. network.receiver.run_step(time=t)

Per-chain isolation is enforced by wiring (each Node has exactly one input
path and one output path), not by data-structure customization. The class
relies on three small backward-compatible upgrades to mp_mh_network/:
  - NodeReceiver.run_step DROPPED-init iterates actual global_path_index
  - NodeSender.perform_natural_matching early-exits when len(paths) <= 1
  - SimSender.run_remaining_paths_and_receiver_step guards
    self.next_hop.run_step() with hasattr (mirroring Node.run_step)
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "jamming_simulation"))

from mp_mh_network.Network import MhNetwork
from mp_mh_network.Node import Node
from mp_mh_network.Receiver import SimReceiver
from mp_mh_network.Sender import SimSender
# Import the leaf modules the SAME way the units (Receiver/Sender/Node) do -- as
# top-level modules (mp_mh_network is on sys.path). Importing them package-qualified
# (mp_mh_network.feedback_source / mp_mh_network.Channels) would create a SECOND copy
# of the module, so FeedbackSource.E2E here would not be identical to the enum member
# the units compare against, breaking the `feedback_source in (...)` assertions.
from Channels import Channel
from feedback_source import FeedbackSource

from JamChannels import JamPath
from Jammer import Jammer


class JamMpMhNetwork(MhNetwork):
    """Multi-hop multipath network of P parallel single-path chains plus a Jammer.

    Sibling of MpMhNetwork (both inherit MhNetwork for the shared per-hop RTT
    split plus Network's stats infrastructure). Does NOT inherit MpMhNetwork
    because the natural-matching update_natural_matching hook does not apply
    here: chain identity is fixed by physical wiring, not by per-step path ranking.
    """

    def __init__(
        self,
        path_epsilons: list[list[float]],   # chain-major: path_epsilons[c][h]
        initial_epsilon: float = None,
        max_iterations: int = None,
        num_packets_to_send: int = None,
        num_paths: int = 4,
        global_prop_delay: int = 6,
        threshold: float = 0.0,
        max_allowed_overlap: int = None,
        num_hops: int = 3,
        jammer_alpha: int = 2,
        jammer_k: int = 1,
        debug: bool = False,
        feedback_source: FeedbackSource = FeedbackSource.HBH,
    ):
        super().__init__(
            path_epsilons,
            initial_epsilon,
            max_iterations,
            num_packets_to_send,
            num_paths,
            global_prop_delay,
            threshold,
            max_allowed_overlap,
            num_hops,
            debug,
        )
        # num_hops / num_nodes / hop_rtt / hop_prop_delay come from MhNetwork.
        assert num_paths >= 1, f"num_paths must be >= 1, got {num_paths}"
        assert len(path_epsilons) == num_paths, (
            f"path_epsilons must be chain-major with len == num_paths "
            f"({num_paths}), got {len(path_epsilons)}"
        )
        for c, chain_eps in enumerate(path_epsilons):
            assert len(chain_eps) == num_hops, (
                f"path_epsilons[{c}] must have num_hops ({num_hops}) entries, "
                f"got {len(chain_eps)}"
            )

        # Feedback source (hop-by-hop vs end-to-end)
        self.feedback_source = feedback_source
        assert feedback_source in (FeedbackSource.HBH, FeedbackSource.E2E), \
            f"Invalid feedback source: {feedback_source}"

        # End-to-end feedback channels: one dedicated channel per global path
        # (= per chain, since chain identity is fixed here), carrying ACK/NACKs
        # straight from the SimReceiver back to the SimSender with the full
        # end-to-end one-way delay (global_prop_delay).
        self.e2e_feedback_channels: dict[int, Channel] = {}
        if self.feedback_source == FeedbackSource.E2E:
            for global_path_idx in range(1, num_paths + 1):
                channel = Channel(
                    self.global_prop_delay,
                    hop_index=0,
                    path_index_in_hop=global_path_idx - 1,
                    name_prefix=f"E2E[{global_path_idx}].",
                    debug=self.debug,
                )
                channel.set_global_path_index(global_path_idx)
                self.e2e_feedback_channels[global_path_idx] = channel

        self.paths: list[list[JamPath]] = [[] for _ in range(num_paths)]
        self._build_paths()

        self.receiver = SimReceiver(
            input_paths=self._paths_at_hop(num_hops - 1),
            hop_rtt=self.hop_rtt,
            unit_name="SimReceiver",
            debug=self.debug,
            feedback_source=self.feedback_source,
            e2e_feedback_channels=self.e2e_feedback_channels,
            e2e_prop_delay=self.global_prop_delay,
        )

        self.nodes: list[list[Node]] = [
            [None] * self.num_nodes for _ in range(num_paths)
        ]
        self._build_nodes()

        init_eps = initial_epsilon if initial_epsilon is not None else 0.0
        self.sender = SimSender(
            num_of_packets_to_send=self.num_packets_to_send,
            global_rtt=self.global_rtt,
            hop_rtt=self.hop_rtt,
            paths=self._paths_at_hop(0),
            initial_epsilon=init_eps,
            max_allowed_overlap=max_allowed_overlap,
            threshold=threshold,
            network=self,
            next_hop=None,
            debug=self.debug,
            feedback_source=self.feedback_source,
            e2e_feedback_channels=self.e2e_feedback_channels,
        )

        all_paths = [p for chain in self.paths for p in chain]
        self.jammer = Jammer(
            paths=all_paths,
            rtt=self.global_rtt,
            alpha=jammer_alpha,
            k=jammer_k,
            parent_network=self,
            debug=self.debug,
        )

    
    def _build_paths(self):
        for c in range(self.num_paths):
            for h in range(self.num_hops):
                jpath = JamPath(
                    propagation_delay=self.hop_prop_delay,
                    epsilon=self.path_epsilons[c][h],
                    hop_index=h,
                    path_index_in_hop=c,
                    debug=self.debug,
                )
                jpath.set_global_path_index(c + 1)
                self.paths[c].append(jpath)
    
    def _build_nodes(self):
        for c in range(self.num_paths):
            for h in range(self.num_nodes):
                node = Node(
                    hop_num=h + 1,
                    input_paths=[self.paths[c][h]],
                    output_paths=[self.paths[c][h + 1]],
                    hop_rtt=self.hop_rtt,
                    unit_name=f"Node[c={c},h={h}]",
                    next_hop=None,
                    Network=self,
                    debug=self.debug,
                    feedback_source=self.feedback_source,
                )
                self.nodes[c][h] = node
    
    def _paths_at_hop(self, h: int) -> list[JamPath]:
        return [self.paths[c][h] for c in range(self.num_paths)]

    def _tick(self):
        """One time step with explicit ordering: jammer -> sender -> nodes (hop-major) -> receiver."""
        self.jammer.run_step(time=self.t)
        self.sender.run_step(time=self.t)
        for h in range(self.num_nodes):
            for c in range(self.num_paths):
                self.nodes[c][h].run_step(time=self.t)
        self.receiver.run_step(time=self.t)

    def run_sim(self):
        if self.max_iterations is not None:
            for t in range(1, self.max_iterations + 1):
                self.t = t
                self._tick()
                if (
                    len(self.receiver.information_packets_decoding_times)
                    >= self.num_packets_to_send
                ):
                    break
            self.collect_stats()
            print(f"Simulation completed at t={self.t} - all packets decoded")
        else:
            while (
                len(self.receiver.information_packets_decoding_times)
                < self.num_packets_to_send
            ):
                self.t += 1
                self._tick()
            self.collect_stats()
            print(f"Simulation completed at t={self.t} - all packets decoded")
