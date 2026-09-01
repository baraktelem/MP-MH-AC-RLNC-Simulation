"""SRJamMpMhNetwork: the uncoded SR-ARQ counterpart of jamming_simulation's
JamMpMhNetwork.

It is the multi-hop SR-ARQ network (P independent chains of H hops, hop-by-hop
SRNodes, decoupled per-chain in-order delivery) plus a Jammer that can block any
forward channel before each tick. Topologically it matches JamMpMhNetwork; the
only difference is the protocol running on top (uncoded SR-ARQ instead of
AC-RLNC).

Implementation: SRJamMpMhNetwork subclasses SRMpMhNetwork so it inherits the
full SR-ARQ stack, feedback modes, window / in_order_forwarding /
node_queue_size / packets_per_path knobs, and stats pipeline unchanged. Only
three things change:

  1. _make_path returns a JamPath (a Path whose ForwardChannel a Jammer can
     block) instead of a plain Path -- SRMpMhNetwork.__init__ then wires those
     JamPaths into the sender / nodes / receiver verbatim.
  2. A Jammer is built over all P*H forward paths (the E2E feedback channels are
     deliberately NOT jammed, so feedback stays lossless, exactly like
     JamMpMhNetwork).
  3. _tick runs the jammer first, then the normal SR tick (source -> nodes
     hop-major -> receiver). Jamming a link for a RTT/alpha-slot window is just a
     forced erasure on that link; the existing SR-ARQ recovery handles it with no
     protocol change.

Tick order per slot:
    1. jammer.run_step(time=t)            # (re)selects k paths every RTT/alpha
    2. sender.run_step(time=t)
    3. for h in 0..num_nodes-1:
           for c in 0..P-1: nodes[c][h].run_step(time=t)
    4. receiver.run_step(time=t)
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "jamming_simulation"))

from mp_mh_network.Channels import Path
from mp_mh_network.Network import SimulationStats

from sr_arq.SRNetwork import SRMpMhNetwork

# Import the jammer leaf modules the same way JamNetwork.py does -- as top-level
# modules (jamming_simulation is on sys.path). JamChannels itself imports
# mp_mh_network.Channels package-qualified, so JamPath's base Path is the SAME
# class object SRMpMhNetwork uses; no double-import / isinstance surprises.
from JamChannels import JamPath
from Jammer import Jammer


class SRJamMpMhNetwork(SRMpMhNetwork):
    """Multi-hop multipath SR-ARQ network of P parallel single-path chains plus a
    Jammer. Sibling of jamming_simulation.JamNetwork.JamMpMhNetwork (same P-chain
    topology and explicit jammer-first tick), but uncoded SR-ARQ instead of
    AC-RLNC.

    Accepts every SRMpMhNetwork argument (path_epsilons, num_paths, num_hops,
    global_prop_delay, window, in_order_forwarding, node_queue_size,
    packets_per_path, feedback_mode, ...) plus two jammer knobs:

      - jammer_k:     number of forward paths blocked each jamming round.
      - jammer_alpha: jamming-round length = global_rtt / alpha (a fresh random
                      set of k paths is chosen every round).
    """

    def __init__(self, *args, jammer_alpha: int = 2, jammer_k: int = 1, **kwargs):
        # SRMpMhNetwork.__init__ builds paths via self._make_path (overridden
        # below to JamPath) and wires them into the sender / nodes / receiver.
        super().__init__(*args, **kwargs)

        # Jam every forward link (all P*H JamPaths). The dedicated per-chain E2E
        # feedback channels live in self.e2e_feedback_channels, NOT in self.paths,
        # so the jammer never touches them -- feedback stays lossless.
        all_paths = [p for chain in self.paths for p in chain]
        self.jammer = Jammer(
            paths=all_paths,
            rtt=self.global_rtt,   # jamming_round_time = global_rtt / alpha
            alpha=jammer_alpha,
            k=jammer_k,
            parent_network=self,
            debug=self.debug,
        )

    def _make_path(self, prop_delay: int, epsilon: float, hop_index: int, path_index_in_hop: int) -> Path:
        # A JamPath is a Path whose ForwardChannel drops the packet whenever the
        # Jammer has flagged it jammed this slot.
        return JamPath(prop_delay, epsilon, hop_index, path_index_in_hop, debug=self.debug)

    def _tick(self):
        # Jam first (sets/clears is_jammed on the selected links for this slot),
        # then run the normal SR tick: source -> nodes (hop-major) -> receiver.
        self.jammer.run_step(time=self.t)
        super()._tick()

    def run_sim(self):
        """Run the jammed SR-ARQ simulation until the selected stop trigger fires.

        Stop trigger (same selection as SRMpMhNetwork / sr_arq_mpmh_simulation.py):
          - max_iterations is not None -> fixed horizon of that many slots (stops
            early if the packet target is reached first).
          - max_iterations is None     -> run until num_packets_to_send in-order
            deliveries. num_packets_to_send already reflects packets_per_path * P
            when a per-chain quota was given (SRMpMhNetwork.__init__ sets that), so
            all three triggers (MAX_ITERATIONS / PACKETS_PER_PATH /
            NUM_PACKETS_TO_SEND) are honored here.

        Two deliberate differences from SRMpMhNetwork.run_sim, both because a chain
        can be starved under jamming:
          - No post-horizon drain (draining every admitted packet could loop forever
            when a chain stays jammed); the finite-horizon result is simply censored.
          - Zero-decoded guard: if nothing decoded, publish zeroed stats instead of
            calling collect_stats (which would divide by zero in the delay stats).

        WARNING: with max_iterations=None the run does NOT terminate if the target is
        unreachable (e.g. jammer_k = P*H, or a very heavy k / high-eps point), since
        the target is never delivered -- use a finite max_iterations for such sweeps.
        """
        if self.max_iterations is not None:
            for t in range(1, self.max_iterations + 1):
                self.t = t
                self._tick()
                if (
                    len(self.receiver.information_packets_decoding_times)
                    >= self.num_packets_to_send
                ):
                    break
        else:
            while (
                len(self.receiver.information_packets_decoding_times)
                < self.num_packets_to_send
            ):
                self.t += 1
                self._tick()
        if len(self.receiver.information_packets_decoding_times) == 0:
            # All-jammed / nothing delivered within the horizon: collect_stats would
            # divide by zero in calculate_inorder_delays_stats, so publish an
            # explicit zeroed result instead.
            self.simulation_stats = SimulationStats(
                time_slots=self.t,
                num_information_packets_sent=self.num_packets_to_send,
            )
        else:
            self.collect_stats()
        if self.debug:
            print(f"Simulation completed at t={self.t}")
