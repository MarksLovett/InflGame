#!/usr/bin/env python3
"""
Parallel Monte Carlo Gradient Ascent Script
Run gradient ascent for multiple random initial conditions in parallel
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import time
from tqdm import tqdm
import multiprocessing as mp

from InflGame.adaptive.visualization import Shell
import InflGame.utils.general as general
import InflGame.domains.rd as rd


def monte_carlo_unit_square(number_samples, num_agents, seed=None):
    """Generate random initial positions for Monte Carlo sampling"""
    if seed is not None:
        torch.manual_seed(seed)
    samples = torch.rand(number_samples, num_agents)
    return samples


def run_single_gradient_ascent(args):
    """
    Run gradient ascent for a single initial condition
    This function will be called in parallel
    """
    initial_pos, sample_idx, config = args
    
    try:
        # Create Shell instance for this sample
        vis = Shell(
            num_agents=config['num_agents'],
            agents_pos=initial_pos.clone(),
            parameters=config['parameters'],
            resource_distribution=config['resource_distribution'],
            bin_points=config['bin_points'],
            infl_configs=config['infl_configs'],
            learning_rate_type=config['learning_rate_type'],
            learning_rate=config['learning_rate'],
            time_steps=config['time_steps'],
            fp=config['fp'],
            infl_cshift=config['infl_cshift'],
            cshift=config['cshift'],
            infl_fshift=config['infl_fshift'],
            Q=config['Q'],
            domain_type=config['domain_type'],
            domain_bounds=config['domain_bounds'],
            resource_type=config['resource_type'],
            domain_refinement=config['domain_refinement'],
            tolerance=config['tolerance'],
            tolerated_agents=config['tolerated_agents'],
            ignore_zero_infl=config['ignore_zero_infl']
        )
        
        # Setup and run gradient ascent
        vis.setup_adaptive_env()
        vis.field.gradient_ascent()
        
        # Extract results
        final_position = vis.field.agents_pos.clone()
        converged = len(vis.field.pos_matrix) < config['time_steps']  # Converged if stopped early
        num_steps = len(vis.field.pos_matrix)
        
        # Get final fitness/payoff
        final_fitness = vis.field.payoff_func(final_position).item()
        
        result = {
            'sample_idx': sample_idx,
            'initial_position': initial_pos.numpy(),
            'final_position': final_position.numpy(),
            'final_fitness': final_fitness,
            'converged': converged,
            'num_steps': num_steps,
            'success': True
        }
        
        return result
        
    except Exception as e:
        # Return error information if something goes wrong
        return {
            'sample_idx': sample_idx,
            'initial_position': initial_pos.numpy(),
            'error': str(e),
            'success': False
        }


def setup_resource_distribution():
    """Setup the resource distribution (you can modify this)"""
    bin_points = np.linspace(.001, .999, 100)
    
    # Gaussian symmetric distribution
    resource_parameters_gaussian = [[.1, .1], [.25, .75], [1, 1]]  # [[sd1, sd2,], [mean1,mean2], [factor1,factor2]]
    resource_distribution = rd.resource_distribution_choice(
        bin_points=bin_points,
        resource_type='multi_modal_gaussian_distribution_1D',
        resource_parameters=resource_parameters_gaussian
    )
    
    return bin_points, resource_distribution


def run_parallel_monte_carlo(n_samples=1000, num_agents=4, n_workers=None, save_results=True):
    """
    Run Monte Carlo gradient ascent in parallel
    
    Parameters:
    -----------
    n_samples : int
        Number of random initial conditions to test
    num_agents : int
        Number of agents in the game
    n_workers : int or None
        Number of parallel workers (None uses all available cores)
    save_results : bool
        Whether to save results to file
    """
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    print(f"Running {n_samples} Monte Carlo samples with {n_workers} parallel workers...")
    
    # Setup resource distribution
    bin_points, resource_distribution = setup_resource_distribution()
    
    # Generate Monte Carlo samples
    print("Generating Monte Carlo samples...")
    samples = monte_carlo_unit_square(n_samples, num_agents, seed=42)
    
    # Setup parameters
    parameters = general.agent_parameter_setup(
        num_agents=num_agents,
        infl_type='gaussian',
        setup_type="initial_symmetric_setup",
        reach=0.1
    )
    
    # Configuration dictionary for all runs
    config = {
        'num_agents': num_agents,
        'parameters': parameters,
        'resource_distribution': resource_distribution,
        'bin_points': bin_points,
        'infl_configs': {'infl_type': 'gaussian'},
        'learning_rate_type': 'cosine_annealing',
        'learning_rate': [.001, .001, 80],
        'time_steps': 10000,
        'fp': 0,
        'infl_cshift': False,
        'cshift': 0,
        'infl_fshift': False,
        'Q': None,
        'domain_type': '1d',
        'domain_bounds': [0, 1],
        'resource_type': 'na',
        'domain_refinement': 10,
        'tolerance': 10**-12,
        'tolerated_agents': None,
        'ignore_zero_infl': True
    }
    
    # Prepare arguments for parallel processing
    args_list = [(samples[i], i, config) for i in range(n_samples)]
    
    # Run parallel processing
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_idx = {executor.submit(run_single_gradient_ascent, args): args[1] 
                        for args in args_list}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_idx), total=n_samples, desc="Processing"):
            result = future.result()
            results.append(result)
    
    end_time = time.time()
    print(f"Completed {n_samples} runs in {end_time - start_time:.2f} seconds")
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"Successful runs: {len(successful_results)}/{n_samples}")
    print(f"Failed runs: {len(failed_results)}")
    
    if successful_results:
        converged_runs = [r for r in successful_results if r['converged']]
        print(f"Converged runs: {len(converged_runs)}/{len(successful_results)}")
        
        # Final fitness statistics
        final_fitnesses = [r['final_fitness'] for r in successful_results]
        print(f"Final fitness - Mean: {np.mean(final_fitnesses):.6f}, Std: {np.std(final_fitnesses):.6f}")
        print(f"Final fitness - Min: {np.min(final_fitnesses):.6f}, Max: {np.max(final_fitnesses):.6f}")
    
    # Save results if requested
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"monte_carlo_results_{timestamp}.pkl"
        
        save_data = {
            'results': results,
            'config': config,
            'n_samples': n_samples,
            'num_agents': num_agents,
            'n_workers': n_workers,
            'runtime_seconds': end_time - start_time,
            'timestamp': timestamp
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Results saved to {filename}")
    
    return results, config


def analyze_results(results_file):
    """Analyze saved results and create visualizations"""
    
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    
    results = data['results']
    config = data['config']
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful results to analyze")
        return
    
    # Extract final positions and fitnesses
    final_positions = np.array([r['final_position'] for r in successful_results])
    final_fitnesses = np.array([r['final_fitness'] for r in successful_results])
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Final fitness histogram
    axes[0, 0].hist(final_fitnesses, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Final Fitness')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Final Fitness Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Final positions for each agent
    for agent in range(config['num_agents']):
        axes[0, 1].hist(final_positions[:, agent], bins=30, alpha=0.6, 
                       label=f'Agent {agent+1}', edgecolor='black')
    axes[0, 1].set_xlabel('Final Position')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Final Agent Positions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter plot of first two agents' positions
    axes[1, 0].scatter(final_positions[:, 0], final_positions[:, 1], 
                      c=final_fitnesses, cmap='viridis', alpha=0.6)
    axes[1, 0].set_xlabel('Agent 1 Position')
    axes[1, 0].set_ylabel('Agent 2 Position')
    axes[1, 0].set_title('Final Positions (Agent 1 vs Agent 2)')
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Final Fitness')
    
    # 4. Convergence analysis
    converged = [r['converged'] for r in successful_results]
    num_steps = [r['num_steps'] for r in successful_results]
    
    axes[1, 1].hist(num_steps, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Number of Steps')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Convergence Steps')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"monte_carlo_analysis_{data['timestamp']}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Monte Carlo Analysis ===")
    print(f"Total samples: {len(results)}")
    print(f"Successful runs: {len(successful_results)}")
    print(f"Convergence rate: {np.mean(converged):.1%}")
    print(f"Average steps to convergence: {np.mean(num_steps):.1f}")
    print(f"Final fitness statistics:")
    print(f"  Mean: {np.mean(final_fitnesses):.6f}")
    print(f"  Std:  {np.std(final_fitnesses):.6f}")
    print(f"  Min:  {np.min(final_fitnesses):.6f}")
    print(f"  Max:  {np.max(final_fitnesses):.6f}")
    

if __name__ == "__main__":
    # Example usage
    print("Starting Parallel Monte Carlo Gradient Ascent")
    
    # Run the parallel Monte Carlo simulation
    results, config = run_parallel_monte_carlo(
        n_samples=100,  # Start with a smaller number for testing
        num_agents=4,
        n_workers=4,    # Adjust based on your CPU cores
        save_results=True
    )
    
    # If you want to analyze saved results later, uncomment this:
    # analyze_results("monte_carlo_results_YYYYMMDD_HHMMSS.pkl")