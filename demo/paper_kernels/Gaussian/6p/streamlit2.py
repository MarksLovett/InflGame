
import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import random
from InflGame.adaptive.visualization import Shell
import InflGame.utils.general as general
import InflGame.domains.rd as rd
import streamlit as st
import hickle as hkl

#The resource points
bin_points=np.linspace(.001, .999, 100)
 
#Resource parameters
resource_parameters_gaussian=[[.1,.1],[.25,.75],[1,1]] #[[sd1, sd2,], [mean1,mean2], [factor1,factor2]]
#Resource distribution
resource_distribution2=rd.resource_distribution_choice(bin_points=bin_points,resource_type='multi_modal_gaussian_distribution_1D',resource_parameters=resource_parameters_gaussian)


domain_type='1d'
resource_distribution=resource_distribution2
mean=np.dot(bin_points,resource_distribution)/np.sum(resource_distribution) #mean of the resource distribution




num_agents=3 #number of agents
#int_agents_pos=general.agent_position_setup(num_agents=num_agents,setup_type='paper_default',domain_type=domain_type,domain_bounds=0)
int_agents_pos=torch.tensor([.2,.5,0.1]) #initial agent positions

infl_configs={"infl_type":"gaussian"} # influence type of the agents



parameters=general.agent_parameter_setup(num_agents=num_agents,infl_type=infl_configs["infl_type"],setup_type="initial_symmetric_setup",reach=0.1) # parameters impacting agents reach (their std)
#parameters_custom=np.array([[.1,.2,.3,...]]) #needs to be length num_players


time_steps=1# number steps for the adaptive dynamics


vis=Shell(num_agents=num_agents,agents_pos=int_agents_pos,parameters=parameters,resource_distribution=resource_distribution,bin_points=bin_points, 
infl_configs = {'infl_type': 'gaussian'}, learning_rate_type= 'cosine_annealing', learning_rate= [.0001, .0001, .08], time_steps=time_steps,
fp= 0, infl_cshift= False, cshift = 0, infl_fshift= False, Q = None,
domain_type = '1d', domain_bounds = [0, 1], resource_type = 'na', domain_refinement = 10,
tolerance = 10**-12, tolerated_agents = None,ignore_zero_infl= True)


vis.setup_adaptive_env()
vis.field.gradient_ascent()
og_pos_matrix=vis.field.pos_matrix
og_grad_matrix=vis.field.grad_matrix
vis.agents_pos=int_agents_pos.clone()
vis.field.agents_pos=int_agents_pos.clone()
results = hkl.load(r'demo\\paper_kernels\\Gaussian\\6p\\3p_paths.hkl')
reach_parameters=general.agent_parameter_setup(num_agents=num_agents,infl_type=vis.infl_type,setup_type="parameter_space",reach_start = .03,reach_end = .3,reach_num_points = 20)
reach_id = st.slider('sigma_id', min_value=0, max_value=len(reach_parameters)-1, value=len(reach_parameters)-1, step=1,label_visibility='hidden')
@st.cache_data
def _plot(results,reach,reach_id, figsize=(14, 10), elev=45, azim=45, show_planes=False):
    """Plot the gradient ascent paths in 3D, looking down the x=y=z diagonal.
    Args:
        self: The Shell instance containing the field and parameters.
        results (list): List of dictionaries with path data.
        figsize (tuple): Size of the figure.
        elev (int): Elevation angle for 3D view.
        azim (int): Azimuth angle for 3D view.
        show_planes (bool): Whether to show equality planes.
    """

    # Create matplotlib figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    sigma=reach
    results=results[list(results.keys())[reach_id]]
    # Add the equality planes if requested
    if show_planes:
        # Create a grid for the planes (reduced resolution for speed)
        grid_points = np.linspace(0, 1, 8)
        X, Y = np.meshgrid(grid_points, grid_points)
        
        # x=y plane (grey)
        ax.plot_surface(X, X, Y, color='grey', alpha=0.15)
        
        # y=z plane (black)
        ax.plot_surface(X, Y, Y, color='black', alpha=0.15)
        
        # x=z plane (brown)
        ax.plot_surface(X, Y, X, color='brown', alpha=0.15)

    # Dictionary to track paths by color for legend
    paths_by_color = {}

    # Plot all paths
    if not results:
        print("WARNING: No valid paths to plot!")
    else:
        print(f"Plotting {len(results)} paths...")
        
        for result in results:
            # Add the path
            line = ax.plot(result['path'][:, 0], result['path'][:, 1], result['path'][:, 2], 
                    color=result['color'], linewidth=2.5, alpha=0.8)[0]
            
            # Track this path for legend
            if result['color'] not in paths_by_color:
                paths_by_color[result['color']] = line
            
            # Add starting point marker (red)
            ax.scatter(result['start'][0], result['start'][1], result['start'][2],
                    color='red', s=60, alpha=1.0, zorder=10)
            
            # If converged, add endpoint marker (green)
            if result['converged']:
                ax.scatter(result['end'][0], result['end'][1], result['end'][2],
                        color='green', s=100, alpha=1.0, zorder=10)

    # Add diagonal line x=y=z (make it thicker and more selfible)
    diag = np.linspace(0, 1, 100)
    ax.plot(diag, diag, diag, 'k--', linewidth=4, alpha=1.0, label='x=y=z')

    # Set labels and title
    ax.set_xlabel('Agent 1 Position', fontweight='bold', fontsize=12)
    ax.set_ylabel('Agent 2 Position', fontweight='bold', fontsize=12)
    ax.set_zlabel('Agent 3 Position', fontweight='bold', fontsize=12)
    ax.set_title('Gradient Ascent Paths (View Along Diagonal)', fontweight='bold', fontsize=14)

    # Set view to look directly down the x=y=z line
    ax.view_init(elev=elev, azim=azim)

    # Region descriptions for legend
    region_descriptions = {
        'red': 'Agent 1 < Agent 2 < Agent 3',
        'green': 'Agent 2 < Agent 3 < Agent 1',
        'blue': 'Agent 3 < Agent 1 < Agent 2',
        'orange': 'Agent 2 < Agent 1 < Agent 3',
        'purple': 'Agent 1 < Agent 3 < Agent 2',
        'cyan': 'Agent 3 < Agent 2 < Agent 1',
        'grey': 'Agent 1 = Agent 2',
        'darkblue': 'Agent 2 = Agent 3',
        'brown': 'Agent 1 = Agent 3',
        'black': 'x = y = z (Diagonal)'
    }

    # Build legend - move it outside the plot area
    legend_elements = []
    for color, line in paths_by_color.items():
        legend_elements.append(line)

    # Add legend with better positioning
    if legend_elements:
        ax.legend(legend_elements, 
                [region_descriptions.get(line.get_color(), line.get_color()) for line in legend_elements],
                loc='center left', 
                bbox_to_anchor=(1.05, 0.5),  # Position legend outside plot
                fontsize=10)

    # Set limits and aspect ratio
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])

    # Add text box with current parameter value
    try:
        sigma_value = sigma  # Replace with actual parameter retrieval if needed
        
        # Format the parameter value for display
        if isinstance(sigma_value, (int, float)):
            param_text = f'σ = {sigma_value:.4f}'
        else:
            param_text = f'σ = {sigma_value}'
        
        # Add text box in the upper left corner
        ax.text2D(0.02, 0.98, param_text, 
                    transform=ax.transAxes, 
                    fontsize=14, 
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='lightblue', 
                            alpha=0.8,
                            edgecolor='black'))
    except Exception as e:
        print(f"Warning: Could not display parameter value: {e}")
    return fig
fig = _plot(results,reach_parameters[reach_id][0].item(),reach_id=reach_id, figsize=(14, 10), elev=45, azim=45, show_planes=False)

st.write(fig)