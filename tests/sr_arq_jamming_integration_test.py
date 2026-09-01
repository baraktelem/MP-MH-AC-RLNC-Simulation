"""
Integration tests for SRJamMpMhNetwork (uncoded SR-ARQ under jamming).

Network-level checks (P chains x H hops, one Jammer over all P*H forward links),
the SR-ARQ analog of tests/jamming_integration_test.py:

- SRJAM1: k=0 (no block)           -> nothing is jammed and every target packet is
                                      delivered (the jam wrapper reduces to plain SR).
- SRJAM2: k=P*H (full block)        -> every source send on hop 0 is jammed, no node
                                      ever receives, the receiver decodes nothing,
                                      and the run terminates at max_iterations.
- SRJAM3: k partial (rotating)      -> the jammer rotates its blocked set over time,
                                      the run still completes, and stats are finite
                                      and strictly worse than the k=0 baseline.
- SRJAM4: E2E feedback channels are never jammed (feedback stays lossless).
"""

import os
import sys
import random

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "jamming_simulation"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sr_arq"))

from sr_arq.SRJamNetwork import SRJamMpMhNetwork
from sr_arq.sr_feedback import SRFeedbackMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Paper article rows as fixed chains (chain-major [chain][hop]) at e1=0.1, e2=0.2,
# but the tests that assert exact jam behavior override epsilons to 0.0 so the ONLY
# packet loss is the jammer (no random channel erasures to muddy the counts).
def _lossless_eps(num_paths: int, num_hops: int) -> list[list[float]]:
    return [[0.0 for _ in range(num_hops)] for _ in range(num_paths)]


def _build(num_paths, num_hops, jammer_k, jammer_alpha, num_packets, max_iterations,
           eps=None, window=None, feedback_mode=SRFeedbackMode.HBH):
    if eps is None:
        eps = _lossless_eps(num_paths, num_hops)
    return SRJamMpMhNetwork(
        path_epsilons=eps,
        num_paths=num_paths,
        num_hops=num_hops,
        global_prop_delay=6,          # RTT/2 for RTT=12; divisible by num_hops=3
        num_packets_to_send=num_packets,
        max_iterations=max_iterations,
        window=window,
        feedback_mode=feedback_mode,
        jammer_alpha=jammer_alpha,
        jammer_k=jammer_k,
    )


def _drive(net, max_iterations, target):
    """Manual tick loop (no collect_stats), so the all-jammed case cannot trip
    the zero-decoded ZeroDivisionError. Returns the number decoded."""
    for t in range(1, max_iterations + 1):
        net.t = t
        net._tick()
        if len(net.receiver.information_packets_decoding_times) >= target:
            break
    return len(net.receiver.information_packets_decoding_times)


def _all_paths(net):
    return [p for chain in net.paths for p in chain]


def _total_jammed(paths):
    return sum(len(p.forward_channel.jammed_packets_history) for p in paths)


def _total_passed(paths):
    return sum(len(p.forward_channel.get_channel_history()) for p in paths)


# ---------------------------------------------------------------------------
# SRJAM1: k=0 no block
# ---------------------------------------------------------------------------

def test_SRJAM1_no_block_delivers_all():
    print("\n=== SRJAM1: k=0 (no block) -> nothing jammed, all delivered ===")
    P, H, N = 4, 3, 120
    net = _build(P, H, jammer_k=0, jammer_alpha=2, num_packets=N, max_iterations=5000)

    assert net.jammer.k == 0
    assert net.jammer.jammed_paths == [], "k=0 jammer must block no paths"

    net.run_sim()  # k=0 always decodes, so collect_stats is safe here

    jammed = _total_jammed(_all_paths(net))
    assert jammed == 0, f"k=0 must jam nothing, but {jammed} packets were jammed"

    decoded = len(net.receiver.information_packets_decoding_times)
    assert decoded >= N, f"expected >= {N} decoded, got {decoded}"

    st = net.get_simulation_stats()
    assert st.normalized_throughput > 0 and st.inorder_delay_mean > 0
    print(f"  decoded={decoded} jammed=0 tp={st.normalized_throughput:.3f} "
          f"dmean={st.inorder_delay_mean:.1f}  PASSED")


# ---------------------------------------------------------------------------
# SRJAM2: k=P*H full block
# ---------------------------------------------------------------------------

def test_SRJAM2_full_block_delivers_nothing():
    print("\n=== SRJAM2: k=P*H (full block) -> receiver decodes nothing ===")
    P, H, N = 4, 3, 40
    MAX_IT = 300
    net = _build(P, H, jammer_k=P * H, jammer_alpha=2, num_packets=N, max_iterations=MAX_IT)

    assert len(net.jammer.paths) == P * H
    assert len(net.jammer.jammed_paths) == P * H, "k=P*H must block every forward link"

    decoded = _drive(net, MAX_IT, target=N)
    assert decoded == 0, f"fully jammed network must decode 0, got {decoded}"

    # Hop 0 (source links): the source keeps sending; every send is jammed, so the
    # jammed history is non-empty and nothing passes through.
    hop0 = [net.paths[c][0] for c in range(P)]
    assert _total_jammed(hop0) > 0, "source should have attempted sends on hop 0"
    assert _total_passed(hop0) == 0, "no hop-0 packet may pass when fully jammed"

    # Downstream hops never receive anything, so their senders never transmit.
    downstream = [net.paths[c][h] for c in range(P) for h in range(1, H)]
    assert _total_passed(downstream) == 0, "no packet may reach/leave any downstream hop"
    print(f"  decoded=0 hop0_jammed={_total_jammed(hop0)} "
          f"downstream_passed=0  PASSED")


# ---------------------------------------------------------------------------
# SRJAM3: partial block rotates and still completes
# ---------------------------------------------------------------------------

def test_SRJAM3_partial_block_rotates_and_degrades():
    print("\n=== SRJAM3: partial k -> jammer rotates, run completes, stats degrade ===")
    P, H, N = 4, 3, 150
    K = 4
    MAX_IT = 8000
    # Paper article rows as chains, with real erasures, so this is a realistic run.
    eps = [[0.1, 0.6, 0.3], [0.8, 0.1, 0.1], [0.2, 0.2, 0.7], [0.2, 0.4, 0.2]]

    # k=0 baseline throughput on the same channel/seed for a strict comparison.
    random.seed(12345)
    base = _build(P, H, jammer_k=0, jammer_alpha=2, num_packets=N, max_iterations=MAX_IT,
                  eps=eps, window=6)
    _drive(base, MAX_IT, target=N)
    base.collect_stats()
    base_tp = base.get_simulation_stats().normalized_throughput

    random.seed(12345)
    net = _build(P, H, jammer_k=K, jammer_alpha=2, num_packets=N, max_iterations=MAX_IT,
                 eps=eps, window=6)
    decoded = _drive(net, MAX_IT, target=N)
    assert decoded >= N, f"partial jam (k={K}) should still complete: decoded {decoded}/{N}"
    net.collect_stats()
    st = net.get_simulation_stats()

    # Jammer actually rotated: several reselect rounds, more than one distinct set.
    history = net.jammer.get_jammed_paths_history()
    assert len(history) >= 3, f"jammer should have run multiple rounds, got {len(history)}"
    distinct = {frozenset(id(p) for p in paths) for paths in history.values() if paths}
    assert len(distinct) > 1, "jammer should have used more than one distinct blocked set"

    # Every jammed set had exactly K paths (except the empty init entry).
    for t, paths in history.items():
        assert len(paths) in (0, K), f"round {t} jammed {len(paths)} paths, expected {K}"

    # Jamming strictly hurts: throughput drops vs the k=0 baseline, stats finite.
    assert 0 < st.normalized_throughput < base_tp, (
        f"jammed throughput {st.normalized_throughput:.3f} should be in "
        f"(0, base {base_tp:.3f})"
    )
    assert st.inorder_delay_mean > 0 and st.inorder_delay_max > 0
    print(f"  decoded={decoded} rounds={len(history)} distinct_sets={len(distinct)} "
          f"tp={st.normalized_throughput:.3f} (base {base_tp:.3f})  PASSED")


# ---------------------------------------------------------------------------
# SRJAM4: E2E feedback channels are never jammed
# ---------------------------------------------------------------------------

def test_SRJAM4_e2e_feedback_never_jammed():
    print("\n=== SRJAM4: E2E feedback channels are not in the jammer's path set ===")
    P, H, N = 4, 3, 60
    net = _build(P, H, jammer_k=P * H, jammer_alpha=2, num_packets=N, max_iterations=200,
                 window=2 * (12 - 1), feedback_mode=SRFeedbackMode.E2E_TIMEOUT)

    # The jammer's path set is exactly the P*H forward links...
    assert len(net.jammer.paths) == P * H
    jammer_ids = {id(p) for p in net.jammer.paths}
    # ...and none of the dedicated per-chain end-to-end feedback channels are in it.
    assert net.e2e_feedback_channels, "E2E mode must build per-chain feedback channels"
    for gid, ch in net.e2e_feedback_channels.items():
        assert id(ch) not in jammer_ids, f"E2E feedback channel {gid} must not be jammable"

    # Sanity: a fully-jammed E2E run still terminates and decodes nothing.
    decoded = _drive(net, 200, target=N)
    assert decoded == 0
    print(f"  P*H={P*H} forward links jammable, "
          f"{len(net.e2e_feedback_channels)} E2E channels untouched  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING SR-ARQ JAMMING INTEGRATION TESTS")
    print("=" * 70)

    test_SRJAM1_no_block_delivers_all()
    test_SRJAM2_full_block_delivers_nothing()
    test_SRJAM3_partial_block_rotates_and_degrades()
    test_SRJAM4_e2e_feedback_never_jammed()

    print("\n" + "=" * 70)
    print("ALL SR-ARQ JAMMING INTEGRATION TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
