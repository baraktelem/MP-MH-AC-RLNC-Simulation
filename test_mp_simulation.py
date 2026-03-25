from Network import MPNetwork, SimulationStats
import numpy as np
import matplotlib.pyplot as plt

def plot_stats(results: list[tuple[float, SimulationStats]]):
    """
    Plot 2D graphs of throughput and in-order delays as functions of eps1.
    
    Args:
        results: List of tuples (eps1, SimulationStats)
    """
    # Extract data
    eps1_values = [r[0] for r in results]
    throughput_values = [r[1].normalized_throughput for r in results]
    delay_mean_values = [r[1].inorder_delay_mean for r in results]
    delay_max_values = [r[1].inorder_delay_max for r in results]
    
    # Calculate channel capacity for each eps1 (single channel)
    channel_capacity = [1 - eps1 for eps1 in eps1_values]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ========================================
    # Plot 1: Normalized Throughput
    # ========================================
    axes[0].plot(eps1_values, throughput_values, marker='o', linewidth=2, 
                 markersize=8, color='#1f77b4', label='Throughput')
    axes[0].plot(eps1_values, channel_capacity, marker='', linewidth=2, 
                 linestyle='--', color='#2ca02c', label='Channel Capacity')
    axes[0].set_xlabel('Epsilon 1 (Path 0)', fontsize=12)
    axes[0].set_ylabel('Normalized Throughput', fontsize=12)
    axes[0].set_title('Normalized Throughput vs Erasure Rate', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].set_xlim([min(eps1_values) - 0.05, max(eps1_values) + 0.05])
    
    # ========================================
    # Plot 2: Mean In-Order Delay
    # ========================================
    axes[1].plot(eps1_values, delay_mean_values, marker='s', linewidth=2,
                 markersize=8, color='#ff7f0e', label='Mean Delay')
    axes[1].set_xlabel('Epsilon 1 (Path 0)', fontsize=12)
    axes[1].set_ylabel('Mean In-Order Delay', fontsize=12)
    axes[1].set_title('Mean In-Order Delay vs Erasure Rate', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([min(eps1_values) - 0.05, max(eps1_values) + 0.05])
    
    # ========================================
    # Plot 3: Max In-Order Delay
    # ========================================
    axes[2].plot(eps1_values, delay_max_values, marker='^', linewidth=2,
                 markersize=8, color='#d62728', label='Max Delay')
    axes[2].set_xlabel('Epsilon 1 (Path 0)', fontsize=12)
    axes[2].set_ylabel('Max In-Order Delay', fontsize=12)
    axes[2].set_title('Max In-Order Delay vs Erasure Rate', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([min(eps1_values) - 0.05, max(eps1_values) + 0.05])
    
    # Add overall title
    fig.suptitle('MP MH AC-RLNC Performance Analysis\n(Single Channel)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('mp_performance_2d.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 2D plots saved to: mp_performance_2d.png")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "MP MH AC-RLNC Simulation")
    print("="*70)
    
    
    eps_values = list(np.arange(0.1, 0.9, 0.1)) # 0.1 to 0.8 in steps of 0.1
    NUM_PATHS = 1
    MAX_ITERATIONS = 150
    PROP_DELAY = 10
    THRESHOLD = 0.0
    O_BAR = 2 * NUM_PATHS * (2*PROP_DELAY - 1)
    print(f"Simulation parameters:")
    print(f"  - Propagation delay: {PROP_DELAY} (RTT={PROP_DELAY*2})")
    print(f"  - Threshold: {THRESHOLD}")
    print(f"  - Max allowed overlap: {O_BAR} (=2k)")
    print(f"  - Number of paths: {NUM_PATHS}")
    print(f"  - Maximum iterations: {MAX_ITERATIONS}")

    results = []
    stats_list = []
    
    total_sims = len(eps_values)
    sim_count = 0
    
    for eps1 in eps_values:
        print(f"--------------------------------")
        print(f"Running simulation {sim_count+1}/{total_sims}: eps1={eps1:.1f}")
        print(f"--------------------------------")
        sim_count += 1
        network = MPNetwork(
            path_epsilons=[eps1],
            initial_epsilon=0.5,
            max_iterations=MAX_ITERATIONS,
            max_allowed_overlap=O_BAR,
            num_paths=NUM_PATHS,
            prop_delay=PROP_DELAY,
            threshold=THRESHOLD
        )
        network.run_sim()
        
        # Store eps values with stats
        stats = network.get_simulation_stats()
        stats_list.append((eps1, stats))
        results.append((eps1, stats))
        
        print(f"  Stats: {stats}")
        for path in network.sender.paths:
            path_params = path.get_params()
            path_index = path.get_global_path_index()
            print(f"  SenderPath[{path_index}]:\n\t\
                epsilon_est{path_index+1}={path_params['epsilon_est']:.2f}\n\t\
                r={path_params['r']:.2f}")
    
    print(f"\n{'='*70}")
    print("All simulations completed! Generating plots...")
    print("="*70)
    
    plot_stats(results)
    stats_sorted_throughput = sorted(stats_list, key=lambda x: x[1].normalized_throughput)
    stats_sorted_delay_mean = sorted(stats_list, key=lambda x: x[1].inorder_delay_mean)
    stats_sorted_delay_max = sorted(stats_list, key=lambda x: x[1].inorder_delay_max)
    # print(f"Stats sorted by throughput:\n{stats_sorted_throughput}")
    # print(f"Stats sorted by delay mean:\n{stats_sorted_delay_mean}")
    # print(f"Stats sorted by delay max:\n{stats_sorted_delay_max}")