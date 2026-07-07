import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))

from mp_mh_network.Network import Network
from mp_mh_network.Channels import Path

from sr_arq.SRReceiver import SRReceiver
from sr_arq.SRSender import SRSender, IndependentSRSender


class SRNetwork(Network):
    """Single-hop multipath Selective-Repeat ARQ network (H=1, P paths).

    Sibling of MPNetwork: one SRSender owning all P paths (shared seq stream
    striped across paths) feeding one SRReceiver that reorders globally. Reuses
    Network.run_sim and the full stats pipeline. No intermediate nodes (those
    come with the multi-hop SRMpMhNetwork variant).

    The sender's next_hop is the receiver, so Network.run_sim's loop of
    sender.run_step() cascades into receiver.run_step() each slot (same pattern
    MPNetwork relies on via SimSender).

    independent=False -> shared-window multipath SR ARQ (reroutes retransmits).
    independent=True  -> per-path-independent SR ARQ (static round-robin split,
                         no rerouting); the paper's SR-ARQ baseline.
    """

    def __init__(
        self,
        path_epsilons: list[float],
        initial_epsilon: float = None,
        max_iterations: int = None,
        num_packets_to_send: int = None,
        num_paths: int = 4,
        prop_delay: int = 10,
        threshold: float = 0.0,            # unused by SR; accepted for API parity
        max_allowed_overlap: int = None,   # unused by SR; accepted for API parity
        independent: bool = False,
        window: int = None,                # per-(sender) sliding-window size; None = unbounded
        debug: bool = False,
    ):
        super().__init__(
            path_epsilons,
            initial_epsilon,
            max_iterations,
            num_packets_to_send,
            num_paths,
            prop_delay,
            threshold,
            max_allowed_overlap,
            debug,
        )
        assert len(path_epsilons) == num_paths, (
            f"path_epsilons must have num_paths ({num_paths}) entries, got {len(path_epsilons)}"
        )

        # Paths (plain Path; the jam variant will swap in JamPath).
        self.paths: list[Path] = [
            Path(prop_delay, eps, 0, i, debug=self.debug) for i, eps in enumerate(path_epsilons)
        ]
        for i, path in enumerate(self.paths):
            path.set_global_path_index(i + 1)

        # Units
        self.receiver = SRReceiver(
            input_paths=self.paths,
            rtt=self.rtt,
            unit_name="SRReceiver",
            debug=self.debug,
        )
        init_eps = initial_epsilon if initial_epsilon is not None else 0.0
        sender_cls = IndependentSRSender if independent else SRSender
        self.sender = sender_cls(
            num_of_packets_to_send=self.num_packets_to_send,
            rtt=self.rtt,
            paths=self.paths,
            initial_epsilon=init_eps,
            window=window,
            next_hop=self.receiver,
            debug=self.debug,
        )
