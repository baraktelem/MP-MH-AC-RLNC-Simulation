from enum import Enum, auto


class SRFeedbackMode(Enum):
    """Feedback / relay mode for the multi-hop SR-ARQ network.

    Defined in its own dependency-free leaf module (mirroring
    ``mp_mh_network/feedback_source.py``) so that SRSender / SRReceiver / SRNode /
    SRNetwork can import it without creating an import cycle.

    - HBH: hop-by-hop. The source hears the first node and every SRNode runs the
      full per-hop SR-ARQ (buffer + node-to-node ACK/NACK + per-hop retransmit).
      End-to-end reliability is the composition of reliable links; per-path rate
      is the min-cut (bottleneck hop). This is the original / default behaviour.

    - E2E_FORWARD_ONLY: the paper's Fig. 19 top graph. Intermediate nodes are
      best-effort forwarders (forward each arrival once, no per-hop retransmission,
      no node-to-node feedback). Only the destination sends end-to-end ACK/NACK per
      chain to the source, which owns all retransmission. The forward delay is
      deterministic, so the end-to-end feedback is SLOT-based (a missing arrival at
      slot t refers to the seq the source sent at t - e2e_prop_delay). A packet is
      delivered only if it survives all H hops, so per-path rate is the product of
      the per-hop rates.

    - E2E_FULL_ARQ: mirrors the mp_mh AC-RLNC E2E structure. Intermediate nodes run
      the full per-hop SR-ARQ among themselves and with the receiver (identical to
      HBH nodes); the only change vs HBH is that the source reads feedback
      exclusively from the receiver (never the first node). Because per-hop ARQ
      makes the end-to-end delay variable, the end-to-end feedback is SEQ-based
      selective-repeat (the receiver ACKs seqs it actually received and NACKs
      genuine per-chain gaps); the source retransmits those seqs, rate-limited to
      once per end-to-end RTT so a merely-delayed in-flight packet is not re-sent.
    """

    HBH = auto()
    E2E_FORWARD_ONLY = auto()
    E2E_FULL_ARQ = auto()

    def is_e2e(self) -> bool:
        return self in (SRFeedbackMode.E2E_FORWARD_ONLY, SRFeedbackMode.E2E_FULL_ARQ)
