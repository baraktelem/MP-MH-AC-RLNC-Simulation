"""
Test utilities for controlled packet dropping in channels and paths.

This module provides TestPath and TestForwardChannel classes that allow
deterministic control over which packets are dropped, instead of using
random dropping with epsilon probability.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Channels import ForwardChannel, Path
from Packet import RLNCPacket


class TestForwardChannel(ForwardChannel):
    """
    ForwardChannel with controlled packet dropping for testing.
    
    Instead of random dropping, maintains a set of packet creation_times
    that should be dropped. This allows deterministic testing.
    """
    
    def __init__(self, propagation_delay: int, epsilon: float, hop_index: int, 
                 path_index_in_hop: int, drop_packet_times: set[int] = None):
        """
        Initialize TestForwardChannel.
        
        Args:
            propagation_delay: Propagation delay in time steps
            epsilon: Kept for compatibility (not used for dropping logic)
            hop_index: Hop index in network
            path_index_in_hop: Path index within hop
            drop_packet_times: Set of creation_times for packets to drop
        """
        super().__init__(propagation_delay, epsilon, hop_index, path_index_in_hop)
        self.drop_packet_times = drop_packet_times if drop_packet_times is not None else set()
        self.channel_name = "TestForward" + self.channel_name  # Update name for clarity
    
    def apply_noise_on_single_packet(self, packet: RLNCPacket) -> tuple[RLNCPacket, bool]:
        """
        Override to drop packets based on creation_time instead of random.
        
        Args:
            packet: RLNCPacket to potentially drop
            
        Returns:
            tuple: (packet, dropped) where dropped is True if packet should be dropped
        """
        dropped = False
        # Check if this packet's creation time is in the drop list
        if packet.get_creation_time() in self.drop_packet_times:
            self.dropped_packets.append(packet)
            self.sim_print(f"Packet dropped:\n\t{packet}", packet.get_creation_time())
            dropped = True
        return packet, dropped


class TestPath(Path):
    """
    Path with TestForwardChannel for controlled packet dropping.
    """
    
    def __init__(self, propagation_delay: int, hop_index: int, 
                 path_index_in_hop: int, drop_packet_times: set[int] = None):
        """
        Initialize TestPath.
        
        Args:
            propagation_delay: Propagation delay in time steps
            epsilon: Not used (kept for compatibility)
            hop_index: Hop index in network
            path_index_in_hop: Path index within hop
            drop_packet_times: Set of creation_times for packets to drop
        """
        super().__init__(propagation_delay, epsilon=0.0, hop_index=hop_index, path_index_in_hop=path_index_in_hop)
        # Replace the forward channel with our test version
        self.forward_channel = TestForwardChannel(
            propagation_delay=propagation_delay,
            epsilon=0.0,
            hop_index=hop_index,
            path_index_in_hop=path_index_in_hop,
            drop_packet_times=drop_packet_times
        )
    
    def get_dropped_packets(self):
        """Get list of dropped packets from forward channel."""
        return self.forward_channel.get_dropped_packets()