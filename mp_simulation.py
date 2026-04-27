"""
MP MH AC-RLNC Simulation with Result Persistence

This script runs multipath simulations and can save/load results for later analysis.

Usage:
    1. Run new simulations and save results:
       python mp_simulation.py
       
    2. Load existing results and replot (set LOAD_EXISTING=True in main):
       Edit the LOAD_EXISTING flag in __main__ section
       
    3. Use the standalone plotter:
       python plot_saved_results.py [results_file.pkl]

Configuration:
    - LOAD_EXISTING: Set to True to load existing results instead of running new simulations
    - RESULTS_FILE: Filename for saving/loading results (default: simulation_results.pkl)
    - NUM_ITERATIONS: Number of iterations to run for each parameter combination
"""
from Network import MPNetwork, SimulationStats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
from datetime import datetime

def save_results(results: list[tuple[float, float, SimulationStats]], filename: str = None):
    """
    Save simulation results to a pickle file.
    
    Args:
        results: List of tuples (eps1, eps2, SimulationStats) from all iterations
        filename: Optional filename. If not provided, generates one with timestamp
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"[OK] Results saved to: {filename}")
    return filename

def load_results(filename: str) -> list[tuple[float, float, SimulationStats]]:
    """
    Load simulation results from a pickle file.
    
    Args:
        filename: Path to the pickle file
        
    Returns:
        List of tuples (eps1, eps2, SimulationStats)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file not found: {filename}")
    
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    print(f"[OK] Results loaded from: {filename}")
    print(f"     Total data points: {len(results)}")
    return results

def aggregate_results(results: list[tuple[float, float, SimulationStats]]):
    """
    Aggregate results by (eps1, eps2) to compute mean and std across iterations.
    
    Args:
        results: List of tuples (eps1, eps2, SimulationStats) from all iterations
        
    Returns:
        Dictionary with key (eps1, eps2) and values dict containing mean and std
    """
    from collections import defaultdict
    
    # Group results by (eps1, eps2)
    grouped = defaultdict(lambda: {'throughput': [], 'delay_mean': [], 'delay_max': []})
    
    for eps1, eps2, stats in results:
        key = (eps1, eps2)
        grouped[key]['throughput'].append(stats.normalized_throughput)
        grouped[key]['delay_mean'].append(stats.inorder_delay_mean)
        grouped[key]['delay_max'].append(stats.inorder_delay_max)
    
    # Calculate mean and std for each (eps1, eps2)
    aggregated = {}
    for key, values in grouped.items():
        aggregated[key] = {
            'throughput_mean': np.mean(values['throughput']),
            'throughput_std': np.std(values['throughput']),
            'delay_mean_mean': np.mean(values['delay_mean']),
            'delay_mean_std': np.std(values['delay_mean']),
            'delay_max_mean': np.mean(values['delay_max']),
            'delay_max_std': np.std(values['delay_max'])
        }
    
    return aggregated

def plot_stats(results: list[tuple[float, float, SimulationStats]]):
    """
    Plot 3D graphs of throughput and in-order delays as functions of eps1 and eps2,
    with error bars showing standard deviation across iterations.
    
    Args:
        results: List of tuples (eps1, eps2, SimulationStats) from all iterations
    """
    # Aggregate results to get mean and std
    aggregated = aggregate_results(results)
    
    # Print some statistics about std values
    all_throughput_stds = [v['throughput_std'] for v in aggregated.values()]
    all_delay_mean_stds = [v['delay_mean_std'] for v in aggregated.values()]
    all_delay_max_stds = [v['delay_max_std'] for v in aggregated.values()]
    
    print(f"\nStandard Deviation Statistics:")
    print(f"  Throughput std - min: {np.min(all_throughput_stds):.6f}, max: {np.max(all_throughput_stds):.6f}, mean: {np.mean(all_throughput_stds):.6f}")
    print(f"  Delay mean std - min: {np.min(all_delay_mean_stds):.6f}, max: {np.max(all_delay_mean_stds):.6f}, mean: {np.mean(all_delay_mean_stds):.6f}")
    print(f"  Delay max std  - min: {np.min(all_delay_max_stds):.6f}, max: {np.max(all_delay_max_stds):.6f}, mean: {np.mean(all_delay_max_stds):.6f}")
    print()
    
    # Get unique eps1 and eps2 values
    eps_pairs = sorted(aggregated.keys())
    unique_eps1 = sorted(list(set([k[0] for k in eps_pairs])))
    unique_eps2 = sorted(list(set([k[1] for k in eps_pairs])))
    
    n_eps1 = len(unique_eps1)
    n_eps2 = len(unique_eps2)
    
    # Create grids for mean and std values
    throughput_mean_grid = np.zeros((n_eps1, n_eps2))
    throughput_std_grid = np.zeros((n_eps1, n_eps2))
    delay_mean_mean_grid = np.zeros((n_eps1, n_eps2))
    delay_mean_std_grid = np.zeros((n_eps1, n_eps2))
    delay_max_mean_grid = np.zeros((n_eps1, n_eps2))
    delay_max_std_grid = np.zeros((n_eps1, n_eps2))
    
    for (eps1, eps2), agg_data in aggregated.items():
        i = unique_eps1.index(eps1)
        j = unique_eps2.index(eps2)
        throughput_mean_grid[i, j] = agg_data['throughput_mean']
        throughput_std_grid[i, j] = agg_data['throughput_std']
        delay_mean_mean_grid[i, j] = agg_data['delay_mean_mean']
        delay_mean_std_grid[i, j] = agg_data['delay_mean_std']
        delay_max_mean_grid[i, j] = agg_data['delay_max_mean']
        delay_max_std_grid[i, j] = agg_data['delay_max_std']
    
    # Create meshgrid for plotting
    EPS1, EPS2 = np.meshgrid(unique_eps1, unique_eps2)
    
    # Add channel capacity calculation
    capacity_grid = np.zeros((n_eps1, n_eps2))
    for i, eps1 in enumerate(unique_eps1):
        for j, eps2 in enumerate(unique_eps2):
            # Capacity = sum of (1 - eps) for all 4 channels
            capacity_grid[i, j] = (1 - eps1) + (1 - eps2) + (1 - 0.2) + (1 - 0.8)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))
    
    # ========================================
    # Plot 1: Normalized Throughput with std
    # ========================================
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot mean surface
    surf1 = ax1.plot_surface(EPS1, EPS2, throughput_mean_grid.T, cmap='viridis', 
                             edgecolor='none', alpha=0.7, label='Throughput')
    
    # Plot capacity surface
    ax1.plot_surface(EPS1, EPS2, capacity_grid.T, color='red', 
                    edgecolor='none', alpha=0.3, label='Capacity')
    
    # Add error bars (vertical lines for std) - plot AFTER surfaces so they're visible
    for i, eps1 in enumerate(unique_eps1):
        for j, eps2 in enumerate(unique_eps2):
            mean_val = throughput_mean_grid[i, j]
            std_val = throughput_std_grid[i, j]
            ax1.plot([eps1, eps1], [eps2, eps2], 
                    [mean_val - std_val, mean_val + std_val],
                    color='black', linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel('Epsilon 1 (Path 0)', fontsize=10)
    ax1.set_ylabel('Epsilon 2 (Path 1)', fontsize=10)
    ax1.set_zlabel('Normalized Throughput', fontsize=10)
    ax1.set_zlim(0, 3)  # Set z-axis limits for throughput
    ax1.set_title('Normalized Throughput vs Erasure Rates', fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # ========================================
    # Plot 2: Mean In-Order Delay with std
    # ========================================
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(EPS1, EPS2, delay_mean_mean_grid.T, cmap='plasma',
                             edgecolor='none', alpha=0.7)
    
    # Add error bars - plot AFTER surface so they're visible
    for i, eps1 in enumerate(unique_eps1):
        for j, eps2 in enumerate(unique_eps2):
            mean_val = delay_mean_mean_grid[i, j]
            std_val = delay_mean_std_grid[i, j]
            ax2.plot([eps1, eps1], [eps2, eps2], 
                    [mean_val - std_val, mean_val + std_val],
                    color='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Epsilon 1 (Path 0)', fontsize=10)
    ax2.set_ylabel('Epsilon 2 (Path 1)', fontsize=10)
    ax2.set_zlabel('Mean In-Order Delay', fontsize=10)
    ax2.set_zlim(0, 600)  # Set z-axis limits for mean delay
    ax2.set_title('Mean In-Order Delay vs Erasure Rates', fontsize=12, fontweight='bold')
    ax2.view_init(elev=20, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # ========================================
    # Plot 3: Max In-Order Delay with std
    # ========================================
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(EPS1, EPS2, delay_max_mean_grid.T, cmap='inferno',
                             edgecolor='none', alpha=0.7)
    
    # Add error bars - plot AFTER surface so they're visible
    for i, eps1 in enumerate(unique_eps1):
        for j, eps2 in enumerate(unique_eps2):
            mean_val = delay_max_mean_grid[i, j]
            std_val = delay_max_std_grid[i, j]
            ax3.plot([eps1, eps1], [eps2, eps2], 
                    [mean_val - std_val, mean_val + std_val],
                    color='black', linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Epsilon 1 (Path 0)', fontsize=10)
    ax3.set_ylabel('Epsilon 2 (Path 1)', fontsize=10)
    ax3.set_zlabel('Max In-Order Delay', fontsize=10)
    ax3.set_title('Max In-Order Delay vs Erasure Rates', fontsize=12, fontweight='bold')
    ax3.set_zlim(0, 600)  # Set z-axis limits for max delay
    ax3.view_init(elev=20, azim=45)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # Add overall title
    fig.suptitle('MP MH AC-RLNC Performance Analysis\n(Path 2: eps=0.2, Path 3: eps=0.8)\nAveraged over iterations',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('mp_performance_3d.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 3D plots saved to: mp_performance_3d.png")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print(" "*20 + "MP MH AC-RLNC Simulation")
    print("="*70)
    
    # ========================================
    # Configuration
    # ========================================
    eps_values = list(np.arange(0.1, 0.9, 0.1)) # 0.1 to 0.8 in steps of 0.1
    NUM_PATHS = 4
    # MAX_ITERATIONS = 150
    MAX_ITERATIONS = None
    NUM_PACKETS_TO_SEND = 200
    PROP_DELAY = 10
    THRESHOLD = 0.0
    EPS3 = 0.2
    EPS4 = 0.8
    O_BAR = 2 * NUM_PATHS * (2*PROP_DELAY - 1)
    NUM_ITERATIONS = 150
    
    # Option to load existing results or run new simulations
    LOAD_EXISTING = False  # Set to True to load existing results, False to run new simulations
    RESULTS_FILE = "simulation_results.pkl"  # File to save/load results

    print(f"\nSimulation parameters:")
    print(f"  - Propagation delay: {PROP_DELAY} (RTT={PROP_DELAY*2})")
    print(f"  - Threshold: {THRESHOLD}")
    print(f"  - Max allowed overlap: {O_BAR} (=2k)")
    print(f"  - Number of paths: {NUM_PATHS}")
    # Use the actual Greek epsilon character in the string, not the LaTeX command.
    print(f"  - ε₃={EPS3}, ε₄={EPS4}")
    if MAX_ITERATIONS is not None:
        print(f"  - Maximum iterations: {MAX_ITERATIONS}")
    else:
        print(f"  - Number of packets to send: {NUM_PACKETS_TO_SEND}")
    print(f"  - Number of iterations: {NUM_ITERATIONS}")

    # ========================================
    # Load existing results or run simulations
    # ========================================
    if LOAD_EXISTING and os.path.exists(RESULTS_FILE):
        print(f"\n{'='*70}")
        print("Loading existing results...")
        print("="*70)
        results = load_results(RESULTS_FILE)
    else:
        print(f"\n{'='*70}")
        print("Running new simulations...")
        print("="*70)
        
        results = []
        
        total_sims_per_iteration = len(eps_values) * len(eps_values)
        total_sims = total_sims_per_iteration * NUM_ITERATIONS
        sim_count = 0
        
        print(f"\nTotal simulations to run: {total_sims} ({NUM_ITERATIONS} iterations x {total_sims_per_iteration} parameter combinations)")
        print(f"{'='*70}\n")
        
        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"\n{'='*70}")
            print(f"Starting iteration {iteration}/{NUM_ITERATIONS}")
            print(f"{'='*70}\n")
            
            for eps1 in eps_values:
                for eps2 in eps_values:
                    sim_count += 1
                    print(f"[{sim_count}/{total_sims}] Iteration {iteration}/{NUM_ITERATIONS}: eps1={eps1:.1f}, eps2={eps2:.1f}, eps3={EPS3:.1f}, eps4={EPS4:.1f}")
                    
                    network = MPNetwork(
                        path_epsilons=[eps1, eps2, EPS3, EPS4],
                        initial_epsilon=0.5,
                        max_iterations=MAX_ITERATIONS,
                        num_packets_to_send=NUM_PACKETS_TO_SEND,
                        max_allowed_overlap=O_BAR,
                        num_paths=NUM_PATHS,
                        prop_delay=PROP_DELAY,
                        threshold=THRESHOLD,
                        debug=False,
                    )
                    network.run_sim()
                    
                    # Store eps values with stats from this iteration
                    stats = network.get_simulation_stats()
                    results.append((eps1, eps2, stats))
                    
                    print(f"  → Throughput: {stats.normalized_throughput:.4f}, Mean delay: {stats.inorder_delay_mean:.2f}, Max delay: {stats.inorder_delay_max}")
                    print(f"  Stats: {stats}")
                    
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}/{NUM_ITERATIONS} completed!")
            print(f"{'='*70}")
            
        print(f"\n{'='*70}")
        print(f"All {total_sims} simulations completed!")
        print("="*70)
        
        # Save results for later use
        save_results(results, RESULTS_FILE)
    
    # ========================================
    # Plot results
    # ========================================
    print(f"\n{'='*70}")
    print("Aggregating results and generating plots...")
    print("="*70)

    plot_stats(results)
    
    print(f"\n{'='*70}")
    print("Plots saved successfully!")
    print("="*70)