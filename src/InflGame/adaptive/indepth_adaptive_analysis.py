from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import torch



def gradient_indepth_analysis(initial_positions, parameters, steps):
    """
    In-depth gradient analysis for specific initial positions using torch tensors.
    Returns all outputs as torch tensors for consistency with grad_func_env.
    """
    vis.field.agents_pos = initial_positions.clone()
    vis.field.time_steps = 1
    
    d_ln_f_list = []
    prob_list = []
    infl_list = []
    grad_list = []
    reward_list = []
    pos_list = []
    
    for _ in range(steps):
        pos_list.append(vis.field.agents_pos.clone())
        d_ln_f = vis.field.d_lnf_matrix(parameter_instance=parameters)
        prob = vis.field.prob_matrix(parameter_instance=parameters)
        infl = vis.field.influence_matrix(parameter_instance=parameters)
        grad = vis.field.gradient(parameter_instance=parameters)
        reward = vis.field.reward_F(parameter_instance=parameters)
        
        vis.field.gradient_ascent()
        vis.field.agents_pos = vis.field.pos_matrix[-1]
        
        d_ln_f_list.append(d_ln_f)
        prob_list.append(prob)
        infl_list.append(infl)
        grad_list.append(grad)
        reward_list.append(reward)
    
    # Stack into tensors
    return (torch.stack(d_ln_f_list), 
            torch.stack(prob_list), 
            torch.stack(infl_list), 
            torch.stack(grad_list), 
            torch.stack(reward_list),
            torch.stack(pos_list))


def gradient_indepth_analysis_plot(bin_points, step_id, pos_matrix, prob_matrix, dmatrix, 
                                   infl_matrix, grad_matrix, reward_matrix, fig=None):
    """
    Plot comprehensive 5-panel gradient analysis using torch tensors.
    Converts to numpy only when needed for matplotlib plotting.
    
    Args:
        bin_points: numpy array or torch tensor of bin points
        step_id: int, which time step to highlight
        pos_matrix: torch tensor (steps, num_agents)
        prob_matrix: torch tensor (steps, num_agents, num_bins)
        dmatrix: torch tensor (steps, num_agents, num_bins)
        infl_matrix: torch tensor (steps, num_agents, num_bins)
        grad_matrix: torch tensor (steps, num_agents)
        reward_matrix: torch tensor (steps, num_agents)
        fig: matplotlib Figure object (if None, creates new figure and shows it)
        
    Returns
    -------
        fig: matplotlib Figure object
    """
    # Convert bin_points to numpy if needed
    if torch.is_tensor(bin_points):
        bin_points_np = bin_points.numpy()
    else:
        bin_points_np = bin_points
    
    # Convert resource_distribution to tensor if needed
    if torch.is_tensor(vis.resource_distribution):
        resource_dist = vis.resource_distribution
    else:
        resource_dist = torch.tensor(vis.resource_distribution)
    
    num_agents = prob_matrix.shape[1]
    
    # Extract the specific time step
    prob_step = prob_matrix[step_id]  # (num_agents, num_bins)
    dmatrix_step = dmatrix[step_id]  # (num_agents, num_bins)
    infl_step = infl_matrix[step_id]  # (num_agents, num_bins)
    grad_step = grad_matrix[step_id]  # (num_agents,)  
    pos_step = pos_matrix[step_id]  # (num_agents,)
    
    # Compute agent-specific resource distributions (torch operations)
    agent_0_resources = prob_step[0] * (1 - prob_step[0]) * resource_dist
    agent_1_resources = prob_step[1] * (1 - prob_step[1]) * resource_dist
    agent_2_resources = prob_step[2] * (1 - prob_step[2]) * resource_dist
    
    # Normalize d_ln_f for visualization
    agent_0_d = dmatrix_step[0] / torch.max(torch.abs(dmatrix_step[0])) / 10
    agent_1_d = dmatrix_step[1] / torch.max(torch.abs(dmatrix_step[1])) / 10
    agent_2_d = dmatrix_step[2] / torch.max(torch.abs(dmatrix_step[2])) / 10
    
    # Create or clear figure
    show_plot = False
    if fig is None:
        fig = plt.figure(constrained_layout=True, figsize=(16, 24))
        show_plot = True
    else:
        fig.clear()
    
    gs = GridSpec(3, 2, figure=fig,width_ratios=[1, 1], wspace=.1,
                        hspace=.1,top=.5)
    
    # Plot 1: Agent resource distributions and d_ln_f
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_box_aspect(1)
    ax1.plot(bin_points_np, agent_0_resources.numpy(), label='Agent 1')
    ax1.plot(bin_points_np, agent_1_resources.numpy(), label='Agent 2')
    ax1.plot(bin_points_np, agent_2_resources.numpy(), label='Agent 3')
    ax1.plot(bin_points_np, agent_0_d.numpy(), c='tab:blue')
    ax1.plot(bin_points_np, agent_1_d.numpy(), c='tab:orange')
    ax1.plot(bin_points_np, agent_2_d.numpy(), c='tab:green')
    ax1.scatter([pos_step[i].item() for i in range(3)],[-.02]*3, c=['tab:blue', 'tab:orange', 'tab:green'], s=100, zorder=5, marker='x', label='Agent positions')
    
    # Compute discrete means using torch
    mean_0 = torch.sum(torch.tensor(bin_points_np) * agent_0_resources) / torch.sum(agent_0_resources)
    mean_1 = torch.sum(torch.tensor(bin_points_np) * agent_1_resources) / torch.sum(agent_1_resources)
    mean_2 = torch.sum(torch.tensor(bin_points_np) * agent_2_resources) / torch.sum(agent_2_resources)
    
    ax1.scatter([mean_0.item(), mean_1.item(), mean_2.item()], [0, 0, 0],c=['tab:blue', 'tab:orange', 'tab:green'], label='Discrete Means')
    ax1.hlines(0, 0, bin_points_np[-1], colors='k', linestyles='dashed')
    ax1.set_title('Agent resource distributions and d_ln_f')
    ax1.legend()
    
    # Plot 2: Agent influence distributions
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_box_aspect(1)
    ax2.plot(bin_points_np, infl_step[0].numpy(), label='Agent 1')
    ax2.plot(bin_points_np, infl_step[1].numpy(), label='Agent 2')
    ax2.plot(bin_points_np, infl_step[2].numpy(), label='Agent 3')
    ax2.set_title('Agent influence distributions')
    ax2.legend()

    # Plot 2: Agent influence distributions
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_box_aspect(1)
    ax2.plot(bin_points_np, prob_step[0].numpy(), label='Agent 1')
    ax2.plot(bin_points_np, prob_step[1].numpy(), label='Agent 2')
    ax2.plot(bin_points_np, prob_step[2].numpy(), label='Agent 3')
    ax2.set_title('Agent Probability distributions')
    ax2.legend()
    
    
    # Plot 3: Agent gradient distributions
    ax3 = fig.add_subplot(gs[1, 1])
    num_steps = pos_matrix.shape[0]
    steps_range = range(num_steps)
    grad_step_list = [grad_matrix[step_id, i] for i in range(reward_matrix.shape[1])]
    ax3.set_box_aspect(1)
    ax3.plot(steps_range, grad_matrix[:, 0].numpy(), label='Agent 1')
    ax3.plot(steps_range, grad_matrix[:, 1].numpy(), label='Agent 2')
    ax3.plot(steps_range, grad_matrix[:, 2].numpy(), label='Agent 3')
    ax3.scatter([step_id]*len(grad_step_list), grad_step_list, 
                label='Agent positions', c=['tab:blue', 'tab:orange', 'tab:green'], s=100)
    ax3.set_title('Agent gradient contributions')
    ax3.legend()
    
    # Plot 4: Agent reward distributions
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_box_aspect(1)
    reward_step_list = [reward_matrix[step_id, i] for i in range(reward_matrix.shape[1])]
    ax4.plot(steps_range, reward_matrix[:, 0].numpy(), label='Agent 1')
    ax4.plot(steps_range, reward_matrix[:, 1].numpy(), label='Agent 2')
    ax4.plot(steps_range, reward_matrix[:, 2].numpy(), label='Agent 3')
    ax4.scatter([step_id]*len(reward_step_list), reward_step_list,
                label='Total rewards', c=['tab:blue', 'tab:orange', 'tab:green'], s=100)
    ax4.set_title('Agent reward distributions')
    ax4.legend()
    
    # Plot 5: Agent position trajectories
    ax5 = fig.add_subplot(gs[0, 1])
    ax5.set_box_aspect(1)
    ax5.plot(steps_range, pos_matrix[:, 0].numpy(), label='Agent 1', c='tab:blue')
    ax5.plot(steps_range, pos_matrix[:, 1].numpy(), label='Agent 2', c='tab:orange')
    ax5.plot(steps_range, pos_matrix[:, 2].numpy(), label='Agent 3', c='tab:green')
    ax5.scatter(step_id, pos_step[0].item(), c='tab:blue', s=100, zorder=5)
    ax5.scatter(step_id, pos_step[1].item(), c='tab:orange', s=100, zorder=5)
    ax5.scatter(step_id, pos_step[2].item(), c='tab:green', s=100, zorder=5)
    ax5.set_title('Agent position trajectories')
    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Position')
    ax5.legend()
    
    if show_plot:
        plt.show()
    
    return fig


def create_gradient_analysis_gif(
    d_ln_f_tensors, 
    prob_tensors, 
    infl_tensors, 
    grad_tensors, 
    reward_tensors, 
    pos_tensors,
    bin_points, 
    resources, 
    num_agents, 
    output_path='gradient_analysis.gif',
    fps=2,
    loop=0,
    dpi=100,
    max_frames=100
):
    """
    Create an animated GIF showing full 5-panel gradient analysis over time steps.
    Reuses gradient_indepth_analysis_plot for consistent rendering.
    
    Parameters
    ----------
    d_ln_f_tensors : torch.Tensor
        Shape (time_steps, num_agents, bins) - derivative of log fitness
    prob_tensors : torch.Tensor
        Shape (time_steps, num_agents, bins) - probability distributions
    infl_tensors : torch.Tensor
        Shape (time_steps, num_agents, bins) - influence values
    grad_tensors : torch.Tensor
        Shape (time_steps, num_agents) - gradient values per agent
    reward_tensors : torch.Tensor
        Shape (time_steps, num_agents) - reward per agent per timestep
    pos_tensors : torch.Tensor
        Shape (time_steps, num_agents) - agent positions per timestep
    bin_points : array-like
        Spatial bin locations
    resources : array-like
        Resource distribution
    num_agents : int
        Number of agents
    output_path : str
        Path to save the GIF file
    fps : int or float
        Frames per second (controls animation speed)
    loop : int
        Number of loops (0 = infinite loop)
    dpi : int
        Resolution of the output GIF
    max_frames : int
        Maximum number of frames to include in GIF (samples evenly from all time steps)
        
    Returns
    -------
    str : path to saved GIF file
    """
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter
    
    total_time_steps = d_ln_f_tensors.shape[0]
    
    # Sample evenly across time steps
    if total_time_steps <= max_frames:
        frame_indices = list(range(total_time_steps))
    else:
        frame_indices = np.linspace(0, total_time_steps - 1, max_frames, dtype=int).tolist()
    
    num_frames = len(frame_indices)
    print(f"Creating GIF with {num_frames} frames from {total_time_steps} time steps (sampling every ~{total_time_steps/num_frames:.1f} steps)")
    
    # Create figure for animation (will be reused for each frame)
    fig = plt.figure(constrained_layout=True, figsize=(15, 10))
    
    def update(frame_idx):
        """Update function for animation - calls gradient_indepth_analysis_plot"""
        time_step = frame_indices[frame_idx]
        
        # Call the gradient_indepth_analysis_plot function with the shared figure
        gradient_indepth_analysis_plot(
            bin_points=bin_points,
            step_id=time_step,
            pos_matrix=pos_tensors,
            prob_matrix=prob_tensors,
            dmatrix=d_ln_f_tensors,
            infl_matrix=infl_tensors,
            grad_matrix=grad_tensors,
            reward_matrix=reward_tensors,
            fig=fig
        )
        
        # Return all axes for animation
        return fig.get_axes()
    
    # Create animation
    print(f"Creating animation at {fps} fps...")
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=num_frames,
        interval=1000/fps,
        blit=False,  # Set to False since we're clearing/redrawing figure
        repeat=True
    )
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    writer = PillowWriter(fps=fps, metadata=dict(artist='InfluenceGame'), bitrate=1800)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    
    print(f"✓ GIF saved to {output_path} ({num_frames} frames at {fps} fps, duration: {num_frames/fps:.1f}s)")
    return output_path



initial_pos = torch.tensor([0.4, 0.5, 0.6])
d_ln_f_tensors, prob_tensors, infl_tensors, grad_tensors, reward_tensors, pos_tensors = gradient_indepth_analysis(initial_pos, parameters, 1200)
# Create a 100-frame GIF from 2000 time steps (samples every ~20 steps)
create_gradient_analysis_gif(
    d_ln_f_tensors, prob_tensors, infl_tensors, grad_tensors, 
    reward_tensors, pos_tensors, bin_points, 
    resources=vis.resource_distribution, 
    num_agents=3,
    output_path='gradient_evolution.gif',
    fps=10,  # 10 frames/second = 10 second GIF
    max_frames=100,  # Sample 100 frames from 2000 time steps
    dpi=100
)