import sys
import os
import math

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))

from mp_mh_network.Network import Network, MhNetwork
from mp_mh_network.Channels import Path, Channel

from sr_arq.SRReceiver import SRReceiver, SRSimReceiver
from sr_arq.SRSender import SRSender, SRSimSender
from sr_arq.SRNode import SRNode
from sr_arq.sr_feedback import SRFeedbackMode


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
            rtt=self.global_rtt,
            unit_name="SRReceiver",
            debug=self.debug,
        )
        init_eps = initial_epsilon if initial_epsilon is not None else 0.0
        sender_cls = SRSimSender if independent else SRSender
        self.sender = sender_cls(
            num_of_packets_to_send=self.num_packets_to_send,
            rtt=self.global_rtt,
            paths=self.paths,
            initial_epsilon=init_eps,
            window=window,
            next_hop=self.receiver,
            debug=self.debug,
        )


class SRMpMhNetwork(MhNetwork):
    """Multi-hop multipath SR-ARQ network: P independent chains of H hops.

    Sibling of JamMpMhNetwork (same P-chain topology and explicit tick order),
    but uncoded hop-by-hop SR-ARQ instead of AC-RLNC:
      - Source = SRSimSender on the hop-0 paths (per-chain round-robin
        stream slices, per-chain sliding window, no rerouting).
      - One SRNode per (chain, hop) doing hop-by-hop store-and-forward ARQ.
      - Receiver = SRSimReceiver with DECOUPLED per-chain in-order delivery.

    With a finite packet target, the simulation stops when all target packets are
    delivered (or max_iterations is reached).  With an infinite packet target
    and finite max_iterations, max_iterations is a fixed measurement horizon:
    throughput is snapshotted at that horizon, source admission is then closed,
    and already-admitted packets are drained before delay statistics are taken.

    packets_per_path (optional): equal new-packet quota per chain. When set,
    each chain admits at most that many new seqs and the global delivery target
    becomes packets_per_path * num_paths. Retransmits are unaffected. Works
    independently of max_iterations (either or both may be set).

    Throughput uses sum(delivered_c / finish_time_c) when equal per-chain packet
    quotas run to completion. If max_iterations binds first, throughput is
    total in-order delivery at that common horizon divided by the horizon; new
    source admission then closes and the admitted cohort drains for uncensored
    D_mean/D_max. H=1 reduces to the single-hop decoupled model.

    path_epsilons is chain-major: path_epsilons[c][h].

    RTT model (global, like MpMhNetwork): global_prop_delay is the END-TO-END
    one-way delay. MhNetwork splits it across the hops into hop_prop_delay
    (= global_prop_delay // num_hops) and hop_rtt (= global_rtt // num_hops), so
    the end-to-end delay is fixed regardless of num_hops. Each hop's Path uses
    hop_prop_delay and every per-hop unit uses hop_rtt.
    """

    def __init__(
        self,
        path_epsilons: list[list[float]],
        initial_epsilon: float = None,
        max_iterations: int = None,
        num_packets_to_send: int = None,
        num_paths: int = 4,
        global_prop_delay: int = 6,
        threshold: float = 0.0,            # unused by SR; accepted for API parity
        max_allowed_overlap: int = None,   # unused by SR; accepted for API parity
        num_hops: int = 3,
        window: int = None,
        in_order_forwarding: bool = False,
        node_queue_size: int = None,
        packets_per_path: int = None,
        feedback_mode: SRFeedbackMode = SRFeedbackMode.HBH,
        debug: bool = False,
    ):
        # Equal per-path quota implies a global delivery target of P * N.
        if packets_per_path is not None:
            assert packets_per_path > 0, (
                f"packets_per_path must be > 0, got {packets_per_path}"
            )
            num_packets_to_send = packets_per_path * num_paths
        # MhNetwork splits the end-to-end global_prop_delay / global_rtt across the
        # hops into self.hop_prop_delay and self.hop_rtt (per-hop quantities),
        # exactly like MpMhNetwork -- this is the "global RTT" model.
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
        self.packets_per_path = packets_per_path
        # Feedback / relay mode (HBH default; two E2E variants). In both E2E modes
        # the source's feedback comes end-to-end from the receiver over dedicated
        # per-chain channels; the relay behaviour differs (see SRFeedbackMode).
        assert feedback_mode in (
            SRFeedbackMode.HBH,
            SRFeedbackMode.E2E_FORWARD_ONLY,
            SRFeedbackMode.E2E_FULL_ARQ,
            SRFeedbackMode.E2E_TIMEOUT,
        ), f"Invalid feedback mode: {feedback_mode}"
        self.feedback_mode = feedback_mode
        # Set only when max_iterations is the binding stop condition. Throughput
        # is snapshotted at this common horizon; the admitted cohort is then
        # drained for uncensored delay statistics.
        self.measurement_horizon: int | None = None
        self.delivered_at_horizon: int | None = None
        self.admitted_at_horizon: frozenset[int] = frozenset()
        self.drain_slots: int = 0
        assert len(path_epsilons) == num_paths, (
            f"path_epsilons must be chain-major with num_paths ({num_paths}) rows, got {len(path_epsilons)}"
        )
        for c, chain in enumerate(path_epsilons):
            assert len(chain) == num_hops, (
                f"path_epsilons[{c}] must have num_hops ({num_hops}) entries, got {len(chain)}"
            )

        # Paths: paths[c][h], each chain uses the same global path index (c+1) at
        # every hop, so a packet's chain is recoverable from its global_path_id.
        self.paths: list[list[Path]] = [[] for _ in range(num_paths)]
        for c in range(num_paths):
            for h in range(num_hops):
                path = Path(self.hop_prop_delay, path_epsilons[c][h], h, c, debug=self.debug)
                path.set_global_path_index(c + 1)
                self.paths[c].append(path)

        # End-to-end feedback channels: one dedicated channel per chain (global
        # path), carrying ACK/NACKs straight from the receiver to the source with
        # the full end-to-end one-way delay. Built for both E2E variants.
        self.e2e_feedback_channels: dict[int, Channel] = {}
        if self.feedback_mode.is_e2e():
            for c in range(num_paths):
                gid = c + 1
                channel = Channel(
                    self.global_prop_delay,
                    hop_index=0,
                    path_index_in_hop=c,
                    name_prefix=f"E2E[{gid}].",
                    debug=self.debug,
                )
                channel.set_global_path_index(gid)
                self.e2e_feedback_channels[gid] = channel

        # Receiver on the last hop of every chain (decoupled per-chain in-order).
        self.receiver = SRSimReceiver(
            input_paths=[self.paths[c][num_hops - 1] for c in range(num_paths)],
            rtt=self.hop_rtt,
            num_chains=num_paths,
            unit_name="SRSimReceiver",
            feedback_mode=self.feedback_mode,
            e2e_feedback_channels=self.e2e_feedback_channels,
            e2e_prop_delay=self.global_prop_delay,
            debug=self.debug,
        )

        # One SRNode per (chain, hop), single-in/single-out; network ticks them.
        self.nodes: list[list[SRNode]] = [[None] * self.num_nodes for _ in range(num_paths)]
        for c in range(num_paths):
            for h in range(self.num_nodes):
                self.nodes[c][h] = SRNode(
                    hop_num=h + 1,
                    input_path=self.paths[c][h],
                    output_path=self.paths[c][h + 1],
                    rtt=self.hop_rtt,
                    unit_name=f"SRNode[c={c},h={h}]",
                    window=window,
                    num_chains=num_paths,
                    in_order_forwarding=in_order_forwarding,
                    node_queue_size=node_queue_size,
                    feedback_mode=self.feedback_mode,
                    debug=self.debug,
                )

        # Source on the hop-0 paths (per-chain independent streams).
        init_eps = initial_epsilon if initial_epsilon is not None else 0.0
        self.sender = SRSimSender(
            num_of_packets_to_send=self.num_packets_to_send,
            rtt=self.hop_rtt,
            paths=[self.paths[c][0] for c in range(num_paths)],
            initial_epsilon=init_eps,
            window=window,
            packets_per_path=packets_per_path,
            next_hop=None,
            feedback_mode=self.feedback_mode,
            e2e_feedback_channels=self.e2e_feedback_channels,
            e2e_rtt=self.global_rtt,
            debug=self.debug,
        )

    def _tick(self):
        # Explicit order: source -> nodes (hop-major) -> receiver.
        self.sender.run_step(time=self.t)
        for h in range(self.num_nodes):
            for c in range(self.num_paths):
                self.nodes[c][h].run_step(time=self.t)
        self.receiver.run_step(time=self.t)

    def run_sim(self):
        if self.max_iterations is not None:
            completed_target = False
            for t in range(1, self.max_iterations + 1):
                self.t = t
                self._tick()
                if len(self.receiver.information_packets_decoding_times) >= self.num_packets_to_send:
                    completed_target = True
                    break

            # If the common horizon bound first, snapshot throughput there,
            # close only NEW source admission, and drain every packet already
            # admitted. Retransmissions and relay forwarding continue normally.
            if not completed_target:
                self.measurement_horizon = self.t
                self.delivered_at_horizon = len(
                    self.receiver.information_packets_decoding_times
                )
                self.admitted_at_horizon = frozenset(
                    self.sender.inforamtion_packets_first_transmission_times
                )
                self.sender.close_admission()

                while not self.admitted_at_horizon.issubset(
                    self.receiver.information_packets_decoding_times
                ):
                    self.t += 1
                    self._tick()
                self.drain_slots = self.t - self.measurement_horizon
        else:
            while len(self.receiver.information_packets_decoding_times) < self.num_packets_to_send:
                self.t += 1
                self._tick()
        self.collect_stats()
        # print(f"Simulation completed at t={self.t}")

    def calculate_normalized_throughput_stats(self):
        if self.measurement_horizon is not None:
            # Fixed-horizon streaming mode: all chains share one wall-clock
            # denominator. Packets delivered during the post-horizon drain are
            # deliberately excluded from throughput.
            self.normalized_throughput = (
                self.delivered_at_horizon / self.measurement_horizon
            )
            return

        # Decoupled: sum of per-chain rates (delivered_c / chain_finish_time_c),
        # used when finite per-chain packet quotas run to completion. A
        # slow/jammed chain does not add artificial idle time to good chains.
        tp = 0.0
        for gid, cnt in self.receiver.chain_delivered_count.items():
            tc = self.receiver.chain_finish_time.get(gid, 0)
            if tc > 0:
                tp += cnt / tc
        self.normalized_throughput = tp
