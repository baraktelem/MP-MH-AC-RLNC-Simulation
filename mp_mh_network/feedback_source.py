from enum import Enum, auto


class FeedbackSource(Enum):
    """Where AC-RLNC feedback (ACK/NACK) originates from.

    Defined in its own leaf module so that Sender/Receiver/Node can import it
    without creating a circular import with Network (which imports all three).
    Network re-exports it, so ``from Network import FeedbackSource`` keeps working.
    """
    E2E = auto()  # End-to-end feedback
    HBH = auto()  # Hop-by-hop feedback
