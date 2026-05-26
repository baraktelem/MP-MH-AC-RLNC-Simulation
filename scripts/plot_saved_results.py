"""
Script to load and plot saved simulation results with comparison support.

Usage:
    # Plot single file:
    python plot_saved_results.py simulation_results.pkl
    
    # Compare multiple protocols:
    python plot_saved_results.py protocol1.pkl protocol2.pkl protocol3.pkl
    
    # Labels can be auto-generated from filenames or you can specify them
"""
import sys
import os
from mp_simulation import load_results, plot_stats, aggregate_results
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def plot_stats_comparison(datasets: list[tuple[str, list]]):
    """
    Plot comparison of multiple protocols on the same 3D graphs.
    
    Args:
        datasets: List of tuples (label, results) where results is list of (eps1, eps2, SimulationStats)
    """
    # Define colors and styles for different protocols
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    colormaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'copper', 'RdPu', 'Greys']
    
    # Get unique eps1 and eps2 values from first dataset (assume all use same grid)
    aggregated_first = aggregate_results(datasets[0][1])
    eps_pairs = sorted(aggregated_first.keys())
    unique_eps1 = sorted(list(set([k[0] for k in eps_pairs])))
    unique_eps2 = sorted(list(set([k[1] for k in eps_pairs])))
    
    n_eps1 = len(unique_eps1)
    n_eps2 = len(unique_eps2)
    
    # Create meshgrid for plotting
    EPS1, EPS2 = np.meshgrid(unique_eps1, unique_eps2)
    
    # Add channel capacity calculation
    capacity_grid = np.zeros((n_eps1, n_eps2))
    for i, eps1 in enumerate(unique_eps1):
        for j, eps2 in enumerate(unique_eps2):
            # Capacity = sum of (1 - eps) for all 4 channels (assuming eps3=0.2, eps4=0.8)
            capacity_grid[i, j] = (1 - eps1) + (1 - eps2) + (1 - 0.2) + (1 - 0.8)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))
    
    # ========================================
    # Plot 1: Normalized Throughput Comparison
    # ========================================
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot capacity surface first (as reference)
    ax1.plot_surface(EPS2, EPS1, capacity_grid.T, color='red', 
                    edgecolor='none', alpha=0.2, label='Capacity')
    
    # Plot each protocol's throughput
    for idx, (label, results) in enumerate(datasets):
        aggregated = aggregate_results(results)
        
        throughput_mean_grid = np.zeros((n_eps1, n_eps2))
        throughput_std_grid = np.zeros((n_eps1, n_eps2))
        
        for (eps1, eps2), agg_data in aggregated.items():
            i = unique_eps1.index(eps1)
            j = unique_eps2.index(eps2)
            throughput_mean_grid[i, j] = agg_data['throughput_mean']
            throughput_std_grid[i, j] = agg_data['throughput_std']
        
        # Plot surface with protocol-specific colormap
        cmap = colormaps[idx % len(colormaps)]
        surf = ax1.plot_surface(EPS2, EPS1, throughput_mean_grid.T, cmap=cmap, 
                                edgecolor='none', alpha=0.6)
        
        # Add error bars
        color = colors[idx % len(colors)]
        for i, eps1 in enumerate(unique_eps1):
            for j, eps2 in enumerate(unique_eps2):
                mean_val = throughput_mean_grid[i, j]
                std_val = throughput_std_grid[i, j]
                ax1.plot([eps2, eps2], [eps1, eps1], 
                        [mean_val - std_val, mean_val + std_val],
                        color=color, linewidth=1.0, alpha=0.7)
    
    ax1.set_xlabel('Epsilon 2 (Path 1)', fontsize=10)
    ax1.set_ylabel('Epsilon 1 (Path 0)', fontsize=10)
    ax1.set_zlabel('Normalized Throughput', fontsize=10)
    ax1.set_zlim(0, 3)  # Set z-axis limits for throughput
    ax1.set_title('Normalized Throughput Comparison', fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    ax1.invert_yaxis()
    
    # ========================================
    # Plot 2: Mean In-Order Delay Comparison
    # ========================================
    ax2 = fig.add_subplot(132, projection='3d')
    
    for idx, (label, results) in enumerate(datasets):
        aggregated = aggregate_results(results)
        
        delay_mean_grid = np.zeros((n_eps1, n_eps2))
        delay_mean_std_grid = np.zeros((n_eps1, n_eps2))
        
        for (eps1, eps2), agg_data in aggregated.items():
            i = unique_eps1.index(eps1)
            j = unique_eps2.index(eps2)
            delay_mean_grid[i, j] = agg_data['delay_mean_mean']
            delay_mean_std_grid[i, j] = agg_data['delay_mean_std']
        
        # Plot surface
        cmap = colormaps[idx % len(colormaps)]
        surf = ax2.plot_surface(EPS2, EPS1, delay_mean_grid.T, cmap=cmap,
                                edgecolor='none', alpha=0.6)
        
        # Add error bars
        color = colors[idx % len(colors)]
        for i, eps1 in enumerate(unique_eps1):
            for j, eps2 in enumerate(unique_eps2):
                mean_val = delay_mean_grid[i, j]
                std_val = delay_mean_std_grid[i, j]
                ax2.plot([eps2, eps2], [eps1, eps1], 
                        [mean_val - std_val, mean_val + std_val],
                        color=color, linewidth=1.0, alpha=0.7)
    
    ax2.set_xlabel('Epsilon 2 (Path 1)', fontsize=10)
    ax2.set_ylabel('Epsilon 1 (Path 0)', fontsize=10)
    ax2.set_zlabel('Mean In-Order Delay', fontsize=10)
    ax2.set_zlim(0, 600)  # Set z-axis limits for mean delay
    ax2.set_title('Mean In-Order Delay Comparison', fontsize=12, fontweight='bold')
    ax2.view_init(elev=20, azim=45)
    ax2.invert_yaxis()
    
    # ========================================
    # Plot 3: Max In-Order Delay Comparison
    # ========================================
    ax3 = fig.add_subplot(133, projection='3d')
    
    for idx, (label, results) in enumerate(datasets):
        aggregated = aggregate_results(results)
        
        delay_max_grid = np.zeros((n_eps1, n_eps2))
        delay_max_std_grid = np.zeros((n_eps1, n_eps2))
        
        for (eps1, eps2), agg_data in aggregated.items():
            i = unique_eps1.index(eps1)
            j = unique_eps2.index(eps2)
            delay_max_grid[i, j] = agg_data['delay_max_mean']
            delay_max_std_grid[i, j] = agg_data['delay_max_std']
        
        # Plot surface
        cmap = colormaps[idx % len(colormaps)]
        surf = ax3.plot_surface(EPS2, EPS1, delay_max_grid.T, cmap=cmap,
                                edgecolor='none', alpha=0.6)
        
        # Add error bars
        color = colors[idx % len(colors)]
        for i, eps1 in enumerate(unique_eps1):
            for j, eps2 in enumerate(unique_eps2):
                mean_val = delay_max_grid[i, j]
                std_val = delay_max_std_grid[i, j]
                ax3.plot([eps2, eps2], [eps1, eps1], 
                        [mean_val - std_val, mean_val + std_val],
                        color=color, linewidth=1.0, alpha=0.7)
    
    ax3.set_xlabel('Epsilon 2 (Path 1)', fontsize=10)
    ax3.set_ylabel('Epsilon 1 (Path 0)', fontsize=10)
    ax3.set_zlabel('Max In-Order Delay', fontsize=10)
    ax3.set_zlim(0, 600)  # Set z-axis limits for max delay
    ax3.set_title('Max In-Order Delay Comparison', fontsize=12, fontweight='bold')
    ax3.view_init(elev=20, azim=45)
    ax3.invert_yaxis()
    
    # Add legend with protocol labels
    legend_text = '\n'.join([f'{colors[i % len(colors)]}: {label}' 
                            for i, (label, _) in enumerate(datasets)])
    
    ## Add overall title with protocol names
    # protocols_str = ' vs '.join([label for label, _ in datasets])
    # fig.suptitle(f'Protocol Comparison: {protocols_str}\n(Path 2: eps=0.2, Path 3: eps=0.8)',
    #              fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('protocol_comparison_3d.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison plots saved to: protocol_comparison_3d.png")
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "Plot Saved Results")
    print("="*70)
    
    # Get filenames from command line or use default
    if len(sys.argv) > 1:
        results_files = sys.argv[1:]
    else:
        current_dir = Path('.')
        mp_sim_results = current_dir / 'simulation_results.pkl'
        sr_sim_results = current_dir / 'sr_arq' / 'results_sr' / 'sr_simulation_results.pkl'
        results_files = [mp_sim_results, sr_sim_results]
        # results_files = [sr_sim_results]
    
    # Check if files exist
    valid_files = []
    for f in results_files:
        if os.path.exists(f):
            valid_files.append(f)
        else:
            print(f"[WARNING] File not found: {f} - skipping")
    
    if not valid_files:
        print("[ERROR] No valid result files found!")
        sys.exit(1)
    
    # Load results from all files
    datasets = []
    print(f"\nLoading {len(valid_files)} result file(s)...")
    print("="*70)
    
    for filepath in valid_files:
        # Extract label from filename (remove path and extension)
        label = os.path.splitext(os.path.basename(filepath))[0]
        
        # Load results
        results = load_results(filepath)
        datasets.append((label, results))
    
    # Plot
    print(f"\n{'='*70}")
    print("Generating plots...")
    print("="*70)
    
    if len(datasets) == 1:
        # Single dataset - use original plotting function
        print("Single protocol - using standard plot")
        plot_stats(datasets[0][1])
    else:
        # Multiple datasets - use comparison plotting function
        print(f"Comparing {len(datasets)} protocols")
        plot_stats_comparison(datasets)
    
    print(f"\n{'='*70}")
    print("Plots saved successfully!")
    print("="*70)

