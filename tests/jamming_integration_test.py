"""
Integration tests for the Jammer with Sender / Node / Receiver.

Test naming convention (SNR total paths = num_paths_per_hop * num_hops):
- JM1: Sender -> Receiver        | 1 path           | k=1 (full block)   -> 0 received, every sent packet is jammed
- JM2: Sender -> Receiver        | 4 paths          | k=4 (full block)   -> 0 received, every sent packet is jammed
- JM3: Sender -> Receiver        | 1 path           | k=0 (no block)     -> all packets decoded, nothing jammed
- JM4: Sender -> Receiver        | 4 paths          | k=2 (partial)      -> per-packet creation_time matches jam_log
- JM5: Sender -> Node -> Receiver| 1 path/hop  (2)  | jam only hop 1, k=1   -> hop 0 delivers, NodeSender sends, hop 1 blocks all
- JM6: Sender -> Node -> Receiver| 4 paths/hop (8)  | jam only hop 1, k=4   -> hop 0 delivers, NodeSender sends, hop 1 blocks all
- JM7: Sender -> Node -> Receiver| 1 path/hop  (2)  | k=0 (no block)        -> all packets decoded end-to-end
- JM8: Sender -> Node -> Receiver| 4 paths/hop (8)  | k=4 (partial, may straddle both hops)
"""

import sys
import os
import random

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mp_mh_network"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "jamming_simulation"))

from Packet import PacketID
from Sender import SimSender
from Receiver import SimReceiver
from Node import Node
from JamChannels import JamPath
from Jammer import Jammer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockMpMhNetwork:
    """Minimal stand-in for MpMhNetwork. Provides what Node/SimSender need
    for natural matching, without bringing the rest of the network class."""

    def __init__(self, num_paths: int):
        self.global_paths_idx_by_r = list(range(1, num_paths + 1))

    def update_natural_matching(self, global_paths_idx_by_r):
        self.global_paths_idx_by_r = global_paths_idx_by_r


def _build_sender_receiver(num_paths: int, num_packets: int, prop_delay: int = 2):
    rtt = prop_delay * 2
    paths = [JamPath(prop_delay, 0.0, 0, i) for i in range(num_paths)]
    for i, path in enumerate(paths):
        path.set_global_path_index(i)
    receiver = SimReceiver(input_paths=paths, rtt=rtt)
    sender = SimSender(
        num_of_packets_to_send=num_packets,
        rtt=rtt,
        paths=paths,
        initial_epsilon=0.0,
        next_hop=receiver,
    )
    return paths, sender, receiver, rtt


def _build_sender_node_receiver(num_paths: int, num_packets: int, prop_delay: int = 2):
    rtt = prop_delay * 2
    paths_hop0 = [JamPath(prop_delay, 0.0, 0, i) for i in range(num_paths)]
    paths_hop1 = [JamPath(prop_delay, 0.0, 1, i) for i in range(num_paths)]
    # MpMhNetwork convention: global path indices are 1-based
    for hop_paths in (paths_hop0, paths_hop1):
        for i, path in enumerate(hop_paths):
            path.set_global_path_index(i + 1)

    network = _MockMpMhNetwork(num_paths)
    node = Node(
        hop_num=1,
        input_paths=paths_hop0,
        output_paths=paths_hop1,
        rtt=rtt,
        Network=network,
    )
    receiver = SimReceiver(input_paths=paths_hop1, rtt=rtt)
    node.next_hop = receiver

    sender = SimSender(
        num_of_packets_to_send=num_packets,
        rtt=rtt,
        paths=paths_hop0,
        initial_epsilon=0.0,
        next_hop=node,
        network=network,
    )
    return paths_hop0, paths_hop1, sender, node, receiver, rtt


def _run(sender, jammer, num_steps: int):
    """Run jammer then sender for num_steps. Sender chains downstream automatically."""
    for t in range(1, num_steps + 1):
        if jammer is not None:
            jammer.run_step(time=t)
        sender.run_step()


def _run_with_jam_log(sender, jammer, num_steps: int) -> dict:
    """Run jammer + sender for num_steps and record, at every t, the set of
    path object-ids the jammer was actively jamming AFTER its run_step
    (i.e. exactly what the sender sees on this step).
    Returns: {t: set(id(path), ...)}."""
    jam_log: dict[int, set[int]] = {}
    for t in range(1, num_steps + 1):
        jammer.run_step(time=t)
        jam_log[t] = {id(p) for p in jammer.jammed_paths}
        sender.run_step()
    return jam_log


def _times_jammed_per_path(jam_log: dict[int, set[int]], paths) -> dict[int, set[int]]:
    """Invert the jam_log: for each path, the set of times it was jammed."""
    per_path: dict[int, set[int]] = {id(p): set() for p in paths}
    for t, jammed_ids in jam_log.items():
        for pid in jammed_ids:
            if pid in per_path:
                per_path[pid].add(t)
    return per_path


def _split_pkts_on_path(history, path, times_jammed: set[int]) -> tuple[set[PacketID], set[PacketID]]:
    """Partition the transmissions in `history` on `path` into the expected
    (blocked_ids, passed_ids) sets based on each packet's creation_time
    and the set of times the path was jammed."""
    target = path.get_global_path_index()
    blocked, passed = set(), set()
    for pkt in history:
        if pkt.get_global_path() != target:
            continue
        if pkt.get_creation_time() in times_jammed:
            blocked.add(pkt.get_id())
        else:
            passed.add(pkt.get_id())
    return set(blocked), set(passed)


def _all_sent_packet_ids(sender) -> set:
    return {pkt.get_id() for pkt in sender.get_all_rlnc_history()}


def _all_jammed_packet_ids(paths) -> set:
    ids = set()
    for path in paths:
        for pkt in path.forward_channel.jammed_packets_history:
            ids.add(pkt.get_id())
    return ids


def _expected_jammed_ids_for(paths_subset, sender) -> set:
    """All sender transmissions whose global_path matches one of paths_subset."""
    target_global_idx = {p.get_global_path_index() for p in paths_subset}
    return {
        pkt.get_id()
        for pkt in sender.get_all_rlnc_history()
        if pkt.get_global_path() in target_global_idx
    }


def _sent_count_on_path(history, path) -> int:
    """Count of RLNC packets in `history` whose global_path == path's.
    The history list should record every attempt regardless of channel outcome
    (SimSender.get_all_rlnc_history() and NodeSender histories both do)."""
    target = path.get_global_path_index()
    return sum(1 for pkt in history if pkt.get_global_path() == target)


def _node_sender_history(node) -> list:
    """Concatenated NodeSender transmissions (NEW + CORRECTION)."""
    return node.my_sender.new_rlnc_packets_history + node.my_sender.correction_packets_history


def _hdr(t: int, name: str) -> str:
    """`[t] name:` prefix matching GeneralUnit.sim_print format."""
    return f"[{t}] {name}:"


def _assert_channel_matches_jammer(path, times_jammed: set[int], path_hdr: str):
    """Verify that JamForwardChannel correctly classifies each packet based
    purely on the Jammer's state.

    Sources used:
      - times_jammed: set of time-steps the Jammer reported this path jammed
        (derived from the jammer's per-step record, not from any sender)
      - path.forward_channel.jammed_packets_history  (channel's own record)
      - path.forward_channel.get_channel_history()   (channel's own record)

    No sender history is consulted -- the SimSender/NodeSender are already
    covered by their own test suites.
    """
    blocked_times = {pkt.get_creation_time() for pkt in path.forward_channel.jammed_packets_history}
    passed_times = {pkt.get_creation_time() for pkt in path.forward_channel.get_channel_history()}

    wrongly_blocked = blocked_times - times_jammed
    wrongly_passed = passed_times & times_jammed

    assert not wrongly_blocked, \
        f"{path_hdr} packets blocked at times when Jammer was NOT jamming this path: " \
        f"{sorted(wrongly_blocked)} (jammed times: {sorted(times_jammed)})"
    assert not wrongly_passed, \
        f"{path_hdr} packets passed through at times when Jammer WAS jamming this path: " \
        f"{sorted(wrongly_passed)} (jammed times: {sorted(times_jammed)})"


# ============================================================================
# JM1-JM4: Sender -> Receiver
# ============================================================================

def test_JM1_SR_one_path_full_block():
    """1 path, k=1: every sent packet must be in jammed_packets_history, and
    the receiver must not see anything."""
    print("\n=== Test JM1: SR | 1 path | k=1 (full block) ===")
    NUM_PATHS, NUM_PACKETS, NUM_STEPS = 1, 10, 30

    paths, sender, receiver, rtt = _build_sender_receiver(NUM_PATHS, NUM_PACKETS)
    jammer = Jammer(paths=paths, rtt=rtt, alpha=1, k=NUM_PATHS)
    _run(sender, jammer, NUM_STEPS)

    assert len(receiver.received_rlnc_channel_history) == 0, \
        f"{_hdr(receiver.t, receiver.unit_name)} Got {len(receiver.received_rlnc_channel_history)} packets, expected 0"

    sent_ids = _all_sent_packet_ids(sender)
    jammed_ids = _all_jammed_packet_ids(paths)
    assert len(sent_ids) > 0, \
        f"{_hdr(sender.t, sender.unit_name)} Sender should have attempted at least 1 send"
    assert sent_ids == jammed_ids, \
        f"{_hdr(sender.t, sender.unit_name)} Sent != Jammed.\n  Missing from jammed: {sent_ids - jammed_ids}\n  Extra in jammed:    {jammed_ids - sent_ids}"

    for path in paths:
        n_sent = _sent_count_on_path(sender.get_all_rlnc_history(), path)
        n_through = len(path.forward_channel.get_channel_history())
        n_jammed = len(path.forward_channel.jammed_packets_history)
        path_hdr = _hdr(sender.t, f"Path[{path.path_index_in_hop}]")
        assert n_sent > 0, \
            f"{path_hdr} sender did not attempt any send on this path"
        assert n_through == 0, \
            f"{path_hdr} should have 0 successful pass-throughs (channel_history) but got {n_through}"
        assert n_jammed == n_sent, \
            f"{path_hdr} jammed_packets_history ({n_jammed}) should equal sender attempts on this path ({n_sent})"

    print(f"  Sent={len(sent_ids)} | Jammed={len(jammed_ids)} | Received=0")
    print("  PASSED")


def test_JM2_SR_four_paths_full_block():
    print("\n=== Test JM2: SR | 4 paths | k=4 (full block) ===")
    NUM_PATHS, NUM_PACKETS, NUM_STEPS = 4, 20, 30

    paths, sender, receiver, rtt = _build_sender_receiver(NUM_PATHS, NUM_PACKETS)
    jammer = Jammer(paths=paths, rtt=rtt, alpha=1, k=NUM_PATHS)
    _run(sender, jammer, NUM_STEPS)

    assert len(receiver.received_rlnc_channel_history) == 0, \
        f"{_hdr(receiver.t, receiver.unit_name)} Got {len(receiver.received_rlnc_channel_history)} packets, expected 0"

    sent_ids = _all_sent_packet_ids(sender)
    jammed_ids = _all_jammed_packet_ids(paths)
    assert len(sent_ids) > 0, \
        f"{_hdr(sender.t, sender.unit_name)} Sender should have attempted at least 1 send"
    assert sent_ids == jammed_ids, \
        f"{_hdr(sender.t, sender.unit_name)} Sent != Jammed.\n  Missing: {sent_ids - jammed_ids}\n  Extra: {jammed_ids - sent_ids}"

    for path in paths:
        n_sent = _sent_count_on_path(sender.get_all_rlnc_history(), path)
        n_through = len(path.forward_channel.get_channel_history())
        n_jammed = len(path.forward_channel.jammed_packets_history)
        path_hdr = _hdr(sender.t, f"Path[{path.path_index_in_hop}]")
        assert n_sent > 0, \
            f"{path_hdr} sender did not attempt any send on this path"
        assert n_through == 0, \
            f"{path_hdr} should have 0 successful pass-throughs (channel_history) but got {n_through}"
        assert n_jammed == n_sent, \
            f"{path_hdr} jammed_packets_history ({n_jammed}) should equal sender attempts on this path ({n_sent})"

    print(f"  Sent={len(sent_ids)} | Jammed={len(jammed_ids)} | Received=0")
    print("  PASSED")


def test_JM3_SR_one_path_no_block():
    """k=0 with 1 path: jammer is present but does not block anything."""
    print("\n=== Test JM3: SR | 1 path | k=0 (no block) ===")
    NUM_PATHS, NUM_PACKETS, NUM_STEPS = 1, 10, 80

    paths, sender, receiver, rtt = _build_sender_receiver(NUM_PATHS, NUM_PACKETS)
    jammer = Jammer(paths=paths, rtt=rtt, alpha=1, k=0)
    _run(sender, jammer, NUM_STEPS)

    for path in paths:
        n_sent = _sent_count_on_path(sender.get_all_rlnc_history(), path)
        n_jammed = len(path.forward_channel.jammed_packets_history)
        n_through = len(path.forward_channel.get_channel_history())
        path_hdr = _hdr(sender.t, f"Path[{path.path_index_in_hop}]")
        assert n_sent > 0, \
            f"{path_hdr} sender did not attempt any send on this path"
        assert n_jammed == 0, \
            f"{path_hdr} should have NO blocked packets but got {n_jammed}"
        assert n_through == n_sent, \
            f"{path_hdr} (eps=0, not jammed) channel_history ({n_through}) should equal sender attempts ({n_sent})"

    expected = set(range(1, NUM_PACKETS + 1))
    decoded = set(receiver.information_packets_decoding_times.keys())
    assert expected.issubset(decoded), \
        f"{_hdr(receiver.t, receiver.unit_name)} Did not decode all packets.\n  Missing: {expected - decoded}"

    print(f"  Decoded {len(decoded & expected)}/{NUM_PACKETS} | Jammed=0")
    print("  PASSED")


def test_JM4_SR_four_paths_partial_block():
    """4 paths, k=2: jammer rotates its 2 jammed paths over time.
    Per-step jam log is captured and used to verify that, for every path,
    jammed_packets_history equals exactly the sender's attempts whose
    creation_time falls in a step where this path was jammed."""
    print("\n=== Test JM4: SR | 4 paths | k=2 (partial block) ===")
    NUM_PATHS, NUM_PACKETS, NUM_STEPS = 4, 2000, 30
    PROP_DELAY = 2
    ALPHA = 1
    K = 2

    paths, sender, receiver, rtt = _build_sender_receiver(NUM_PATHS, NUM_PACKETS, PROP_DELAY)
    random.seed(42)
    jammer = Jammer(paths=paths, rtt=rtt, alpha=ALPHA, k=K)  # jamming_round_time = rtt

    jam_log = _run_with_jam_log(sender, jammer, NUM_STEPS)
    times_jammed = _times_jammed_per_path(jam_log, paths)
    switch_times = set(jammer.get_jammed_paths_history().keys())
    
    expected_times_jammed = set(t for t in range(0, NUM_STEPS + 1, rtt // ALPHA))   
    assert switch_times == expected_times_jammed, f"Times jammed should be at every jamming round time. Times jammed: {switch_times}, expected times jammed: {expected_times_jammed}"
    
    # The jammer should actually rotate within NUM_STEPS (round_time = rtt = 4)
    unique_jam_sets = {frozenset(s) for s in jam_log.values()}
    assert len(unique_jam_sets) > 1, \
        f"{_hdr(jammer.t, jammer.name)} Jammer should have switched its jammed set at least once " \
        f"during the run, but it only used {unique_jam_sets}"

    for path in paths:
        n_sent = _sent_count_on_path(sender.get_all_rlnc_history(), path)
        path_hdr = _hdr(sender.t, f"Path[{path.path_index_in_hop}]")
        assert n_sent > 0, f"{path_hdr} sender did not attempt any send on this path"

        expected_blocked, expected_passed = _split_pkts_on_path(
            sender.get_all_rlnc_history(), path, times_jammed[id(path)]
        )
        actual_blocked = {pkt.get_id() for pkt in path.forward_channel.jammed_packets_history}
        actual_passed = {pkt.get_id() for pkt in path.forward_channel.get_channel_history()}

        assert actual_blocked == expected_blocked, \
            f"{path_hdr} Blocked set mismatch.\n  Missing: {expected_blocked - actual_blocked}" \
            f"\n  Extra: {actual_blocked - expected_blocked}"
        assert actual_passed == expected_passed, \
            f"{path_hdr} Pass-through set mismatch (eps=0 -> all non-jammed sends should pass).\n" \
            f"  Missing: {expected_passed - actual_passed}\n  Extra: {actual_passed - expected_passed}"

    total_blocked = sum(len(path.forward_channel.jammed_packets_history) for path in paths)
    print(f"  Distinct jam sets used: {len(unique_jam_sets)} | Total blocked packets: {total_blocked}")
    print("  PASSED")


# ============================================================================
# JM5-JM8: Sender -> Node -> Receiver
# ============================================================================

def _assert_hop0_unjammed_passthrough(sender, paths_hop0):
    """Hop 0 not jammed, eps=0: every SimSender attempt passes through and
    nothing lands in jammed_packets_history."""
    sim_history = sender.get_all_rlnc_history()
    assert len(sim_history) > 0, \
        f"{_hdr(sender.t, sender.unit_name)} SimSender should have attempted at least 1 send"
    for path in paths_hop0:
        n_sent = _sent_count_on_path(sim_history, path)
        n_through = len(path.forward_channel.get_channel_history())
        n_jammed = len(path.forward_channel.jammed_packets_history)
        path_hdr = _hdr(sender.t, f"Hop0.Path[{path.path_index_in_hop}]")
        assert n_sent > 0, f"{path_hdr} SimSender did not attempt any send on this path"
        assert n_jammed == 0, \
            f"{path_hdr} should have NO blocked packets but got {n_jammed}"
        assert n_through == n_sent, \
            f"{path_hdr} channel_history ({n_through}) should equal SimSender attempts ({n_sent})"


def _assert_full_block_hop1(sender, node, paths_hop1):
    """Hop 1 fully jammed: NodeSender does try to send (hop 0 is delivering),
    and every one of its attempts must be in jammed_packets_history."""
    node_history = _node_sender_history(node)
    assert len(node_history) > 0, \
        f"{_hdr(sender.t, 'NodeSender')} NodeSender should have attempted at least 1 send"
    for path in paths_hop1:
        n_sent = _sent_count_on_path(node_history, path)
        n_through = len(path.forward_channel.get_channel_history())
        n_jammed = len(path.forward_channel.jammed_packets_history)
        path_hdr = _hdr(sender.t, f"Hop1.Path[{path.path_index_in_hop}]")
        assert n_sent > 0, \
            f"{path_hdr} NodeSender did not attempt any send on this path"
        assert n_through == 0, \
            f"{path_hdr} should have 0 successful pass-throughs but got {n_through}"
        assert n_jammed == n_sent, \
            f"{path_hdr} jammed_packets_history ({n_jammed}) should equal NodeSender attempts ({n_sent})"


def test_JM5_SNR_full_block_hop1():
    """1 path per hop (2 total), jammer covers only hop 1 (k=1).
    Hop 0 must deliver normally; the Node must receive packets and try to
    forward them; every NodeSender attempt on hop 1 must be jammed.
    The end receiver gets nothing."""
    print("\n=== Test JM5: SNR | 1 path/hop (total=2) | k=1 (full block on hop 1) ===")
    NUM_PATHS_PER_HOP, NUM_PACKETS, NUM_STEPS = 1, 10000, 40

    paths_hop0, paths_hop1, sender, node, receiver, rtt = \
        _build_sender_node_receiver(NUM_PATHS_PER_HOP, NUM_PACKETS)
    jammer = Jammer(paths=paths_hop1, rtt=rtt, alpha=1, k=NUM_PATHS_PER_HOP)
    _run(sender, jammer, NUM_STEPS)

    nr = node.my_receiver
    assert len(nr.get_received_rlnc_channel_history()) == NUM_STEPS-rtt//2, \
        f"{_hdr(nr.t, nr.unit_name)} Node should receive packets (hop 0 is not jammed) but got {len(nr.get_received_rlnc_channel_history())}. Expected {NUM_STEPS-rtt//2}"
    assert len(receiver.received_rlnc_channel_history) == 0, \
        f"{_hdr(receiver.t, receiver.unit_name)} Receiver should not receive anything when hop 1 is fully jammed"

    _assert_hop0_unjammed_passthrough(sender, paths_hop0)
    _assert_full_block_hop1(sender, node, paths_hop1)

    sent_on_hop1 = len(_node_sender_history(node))
    jammed_on_hop1 = sum(len(p.forward_channel.jammed_packets_history) for p in paths_hop1)
    node_rx = len(node.my_receiver.get_received_rlnc_channel_history())
    print(f"  Node received={node_rx} | NodeSender sent (hop1)={sent_on_hop1} | "
          f"Hop1 jammed={jammed_on_hop1} | Receiver=0")
    print("  PASSED")


def test_JM6_SNR_full_block_hop1_multipath():
    """4 paths per hop (8 total), jammer covers only hop 1 (k=4)."""
    print("\n=== Test JM6: SNR | 4 paths/hop (total=8) | k=4 (full block on hop 1) ===")
    NUM_PATHS_PER_HOP, NUM_PACKETS, NUM_STEPS = 4, 20000, 40

    paths_hop0, paths_hop1, sender, node, receiver, rtt = \
        _build_sender_node_receiver(NUM_PATHS_PER_HOP, NUM_PACKETS)
    jammer = Jammer(paths=paths_hop1, rtt=rtt, alpha=1, k=NUM_PATHS_PER_HOP)
    _run(sender, jammer, NUM_STEPS)

    nr = node.my_receiver
    expected_node_rx = (NUM_STEPS-rtt//2)*NUM_PATHS_PER_HOP 
    assert len(nr.get_received_rlnc_channel_history()) == expected_node_rx, \
        f"{_hdr(nr.t, nr.unit_name)} Node should receive packets (hop 0 is not jammed) but got {len(nr.get_received_rlnc_channel_history())}. Expected {expected_node_rx}"
    assert len(receiver.received_rlnc_channel_history) == 0, \
        f"{_hdr(receiver.t, receiver.unit_name)} Receiver should not receive anything when hop 1 is fully jammed"

    _assert_hop0_unjammed_passthrough(sender, paths_hop0)
    _assert_full_block_hop1(sender, node, paths_hop1)

    sent_on_hop1 = len(_node_sender_history(node))
    jammed_on_hop1 = sum(len(p.forward_channel.jammed_packets_history) for p in paths_hop1)
    node_rx = len(node.my_receiver.get_received_rlnc_channel_history())
    print(f"  Node received={node_rx} | NodeSender sent (hop1)={sent_on_hop1} | "
          f"Hop1 jammed={jammed_on_hop1} | Receiver=0")
    print("  PASSED")


def test_JM7_SNR_no_block():
    """1 path per hop (2 total), k=0: jammer present but blocks nothing.
    All packets must reach the receiver."""
    print("\n=== Test JM7: SNR | 1 path/hop (total=2) | k=0 (no block) ===")
    NUM_PATHS_PER_HOP, NUM_PACKETS, NUM_STEPS = 1, 10, 200

    paths_hop0, paths_hop1, sender, node, receiver, rtt = \
        _build_sender_node_receiver(NUM_PATHS_PER_HOP, NUM_PACKETS)
    all_paths = paths_hop0 + paths_hop1
    jammer = Jammer(paths=all_paths, rtt=rtt, alpha=1, k=0)
    _run(sender, jammer, NUM_STEPS)

    sim_history = sender.get_all_rlnc_history()
    for path in paths_hop0:
        n_sent = _sent_count_on_path(sim_history, path)
        n_jammed = len(path.forward_channel.jammed_packets_history)
        n_through = len(path.forward_channel.get_channel_history())
        path_hdr = _hdr(sender.t, f"Hop0.Path[{path.path_index_in_hop}]")
        assert n_sent > 0, f"{path_hdr} sender did not attempt any send on this path"
        assert n_jammed == 0, f"{path_hdr} should have NO blocked packets but got {n_jammed}"
        assert n_through == n_sent, \
            f"{path_hdr} (eps=0, not jammed) channel_history ({n_through}) should equal sender attempts ({n_sent})"

    node_history = _node_sender_history(node)
    for path in paths_hop1:
        n_sent = _sent_count_on_path(node_history, path)
        n_jammed = len(path.forward_channel.jammed_packets_history)
        n_through = len(path.forward_channel.get_channel_history())
        path_hdr = _hdr(sender.t, f"Hop1.Path[{path.path_index_in_hop}]")
        assert n_sent > 0, f"{path_hdr} NodeSender did not attempt any send on this path"
        assert n_jammed == 0, f"{path_hdr} should have NO blocked packets but got {n_jammed}"
        assert n_through == n_sent, \
            f"{path_hdr} (eps=0, not jammed) channel_history ({n_through}) should equal NodeSender attempts ({n_sent})"

    expected = set(range(1, NUM_PACKETS + 1))
    decoded = set(receiver.information_packets_decoding_times.keys())
    assert expected.issubset(decoded), \
        f"{_hdr(receiver.t, receiver.unit_name)} Missing decoded packets: {expected - decoded}"

    print(f"  Decoded {len(decoded & expected)}/{NUM_PACKETS} | Jammed=0")
    print("  PASSED")


def test_JM8_SNR_partial_block():
    """4 paths per hop (8 total), k=4: jammer reselects EVERY step
    (alpha=rtt -> jamming_round_time=1) so each step toggles a fresh half
    of the network's paths.

    Verification uses ONLY:
      - the Jammer's per-step record of jammed paths (jam_log)
      - each JamForwardChannel's own histories
    No sender history is consulted -- SimSender / NodeSender are covered
    by their own test suites."""
    print("\n=== Test JM8: SNR | 4 paths/hop (total=8) | k=4 (reselect every step) ===")
    NUM_PATHS_PER_HOP, NUM_PACKETS, NUM_STEPS = 4, 20, 40
    NUM_HOPS = 2
    TOTAL_PATHS = NUM_PATHS_PER_HOP * NUM_HOPS  # 8
    K = TOTAL_PATHS // 2  # 4
    PROP_DELAY = 2

    paths_hop0, paths_hop1, sender, node, receiver, rtt = \
        _build_sender_node_receiver(NUM_PATHS_PER_HOP, NUM_PACKETS, PROP_DELAY)
    all_paths = paths_hop0 + paths_hop1
    random.seed(42)
    ALPHA = rtt  # jamming_round_time = rtt / alpha = 1 -> max transitions
    jammer = Jammer(paths=all_paths, rtt=rtt, alpha=ALPHA, k=K)

    jam_log = _run_with_jam_log(sender, jammer, NUM_STEPS)
    times_jammed = _times_jammed_per_path(jam_log, all_paths)

    # Reselects happen at every multiple of jamming_round_time (key 0 = __init__).
    step = int(rtt / ALPHA)
    expected_reselect_times = set(range(0, NUM_STEPS + 1, step))
    actual_reselect_times = set(jammer.get_jammed_paths_history().keys())
    assert actual_reselect_times == expected_reselect_times, \
        f"{_hdr(jammer.t, jammer.name)} Reselect times mismatch.\n" \
        f"  Expected (every jamming_round_time={step}): {sorted(expected_reselect_times)}\n" \
        f"  Actual                                    : {sorted(actual_reselect_times)}"

    # With alpha=rtt and many reselects, the active set should vary a lot.
    unique_jam_sets = {frozenset(s) for s in jam_log.values()}
    assert len(unique_jam_sets) > 1, \
        f"{_hdr(jammer.t, jammer.name)} Jammer should have switched its jammed set at least once " \
        f"during the run, but it only used {unique_jam_sets}"

    # Per-path: each channel's two histories must agree with the Jammer's record.
    for hop_idx, hop_paths in ((0, paths_hop0), (1, paths_hop1)):
        for path in hop_paths:
            path_hdr = _hdr(sender.t, f"Hop{hop_idx}.Path[{path.path_index_in_hop}]")
            _assert_channel_matches_jammer(path, times_jammed[id(path)], path_hdr)

    total_blocked = sum(len(p.forward_channel.jammed_packets_history) for p in all_paths)
    total_passed = sum(len(p.forward_channel.get_channel_history()) for p in all_paths)
    print(f"  Distinct jam sets used: {len(unique_jam_sets)} | "
          f"Blocked={total_blocked} | Passed={total_passed}")
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING JAMMER INTEGRATION TESTS")
    print("=" * 70)

    test_JM1_SR_one_path_full_block()
    test_JM2_SR_four_paths_full_block()
    test_JM3_SR_one_path_no_block()
    test_JM4_SR_four_paths_partial_block()
    test_JM5_SNR_full_block_hop1()
    test_JM6_SNR_full_block_hop1_multipath()
    test_JM7_SNR_no_block()
    test_JM8_SNR_partial_block()

    print("\n" + "=" * 70)
    print("ALL JAMMER INTEGRATION TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
