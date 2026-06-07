r"""
.. module:: MARL_plots
   :synopsis: Provides visualization tools for analyzing multi-agent reinforcement learning (MARL) performance in influencer games.

Multi-Agent Reinforcement Learning (MARL) Plotting Module
=========================================================

This module provides visualization tools for analyzing the performance of multi-agent reinforcement learning (MARL) 
algorithms. It includes functions for plotting policies, rewards, and positions of agents over time.

Dependencies:
-------------
- InflGame.utils
- InflGame.MARL

Usage:
------
The `policy_histogram` function visualizes the Q-table as a policy heatmap, while the `reward_plot` and `pos_plot` 
functions plot the rewards and positions of agents over time, respectively. The `policy_deterministically_to_actions` 
function simulates deterministic actions for agents based on their policies.

Example:
--------

.. code-block:: python

    import numpy as np
    import torch
    from InflGame.MARL.async_game import influencer_env_async
    from InflGame.MARL.MARL_plots import policy_histogram, reward_plot, pos_plot, policy_deterministically_to_actions

    # Define environment configuration
    env_config = {
        "num_agents": 3,
        "initial_position": [0.2, 0.5, 0.8],
        "bin_points": np.linspace(0, 1, 100),
        "resource_distribution": np.random.rand(100),
        "step_size": 0.01,
        "domain_type": "1d",
        "domain_bounds": [0, 1],
        "infl_configs": {"infl_type": "gaussian"},
        "parameters": [0.1, 0.1, 0.1],
        "fixed_pa": 0,
        "NUM_ITERS": 100
    }

    # Initialize the environment
    env = influencer_env_async(config=env_config)

    # Simulate deterministic actions
    q_tensor = torch.rand((3, 100, 3))  # Example Q-tensor
    pos_matrix, reward_matrix = policy_deterministically_to_actions(env=env, q_tensor=q_tensor, num_step=50)

    # Plot policy heatmap for player 0
    policy_fig = policy_histogram(q_tensor=q_tensor, player_id=0)
    policy_fig.show()

    # Plot rewards over time
    reward_fig = reward_plot(reward_matrix=reward_matrix, possible_agents=env.possible_agents)
    reward_fig.show()

    # Plot positions over time
    pos_fig = pos_plot(pos_matrix=pos_matrix, possible_agents=env.possible_agents, domain_bounds=env_config["domain_bounds"])
    pos_fig.show()


"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure
import torch
from typing import Type, TypeVar
from matplotlib import colors as mpl_colors
import matplotlib as mpl
from typing import List, Tuple, Dict, Optional

import InflGame.utils.general as general
import InflGame.MARL.utils.IQL_utils as IQL_utils
from InflGame.MARL.sync_game import influencer_env_sync
from InflGame.MARL.async_game import influencer_env_async
from InflGame.domains.one_d import one_utils
from InflGame.utils import data_management


def policy_histogram(num_agents=2,
                     q_table: dict = None,
                     q_tensor: torch.Tensor = None,
                     agent_id: int = 0,
                     temperature: float = 1,
                     title_ads: List[str] = [],
                     name_ads: List[str] = [],
                     save: bool = False,
                     save_types: List[str] = ['.png', '.svg'],
                     font: dict = {'default_size': 12, 'title_size': 14, 'legend_size': 12, 'font_family': 'sans-serif'},
                     paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'policy_histogram'},
                     figsize: Tuple[float, float] = (6, 6),
                     axis_return: bool = False,
                     ) -> matplotlib.figure.Figure:
    r"""
    Visualizes the Q-table as a policy using a softmax function and plots it as a heatmap.

    .. math::
        P(a|s) = \frac{e^{Q(s,a)/T}}{\sum_{a'} e^{Q(s,a')/T}}

    where:
      - :math:`a` is the action 
      - :math:`s` is the current state
      - :math:`a'` is the next state 
      - :math:`T` is the temperature parameter
      - :math:`P(a|s)` is the probability of taking action :math:`a` in state :math:`s`
      - :math:`Q(s,a)` is the Q-value for action :math:`a` in state :math:`s`        

    :param q_table: Q-table in dictionary format. Defaults to None.
    :type q_table: dict, optional
    :param q_tensor: Q-table as a torch.Tensor. Defaults to None.
    :type q_tensor: torch.Tensor, optional
    :param agent_id: Agent's ID number. Defaults to 0.
    :type agent_id: int
    :param temperature: A smoothness factor for the softmax function. Defaults to 1.
    :type temperature: float
    :param title_ads: List of additional title components to append. Default is ``[]``.
    :type title_ads: List[str], optional
    :param name_ads: List of additional name components for file saving. Default is ``[]``.
    :type name_ads: List[str], optional
    :param save: If ``True``, saves the figure to file. Default is ``False``.
    :type save: bool, optional
    :param save_types: List of file extensions for saving. Default is ``['.png', '.svg']``.
    :type save_types: List[str], optional
    :param font: Font configuration dictionary with keys ``'default_size'``, ``'title_size'``,
        ``'legend_size'``, and ``'font_family'``.
    :type font: dict, optional
    :param paper_figure: Paper figure configuration with keys ``'paper'``, ``'section'``,
        and ``'figure_id'``.
    :type paper_figure: dict, optional
    :param figsize: Figure size as (width, height). Default is ``(6, 6)``.
    :type figsize: Tuple[float, float], optional
    :param axis_return: If ``True``, returns the axes object instead of figure. Default is ``False``.
    :type axis_return: bool, optional
    :return: Figure representing the policy as a heatmap (or Axes if ``axis_return=True``).
    :rtype: matplotlib.figure.Figure or matplotlib.axes.Axes
    """
    font_family = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)

    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font_family})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    if q_table is not None:
        q_tensor = IQL_utils.Q_table_to_tensor(q_table)
    fig, ax = plt.subplots(figsize=figsize)
    policy = IQL_utils.Q_tensor_to_policy(q_tensor=q_tensor, agent_id=agent_id, temperature=temperature)
    ax = sns.heatmap(policy, xticklabels=['left', 'stay', 'right'], yticklabels=5, cmap=sns.cubehelix_palette(as_cmap=True), vmax=1, vmin=0, ax=ax)
    ax.invert_yaxis()
    ax.set_box_aspect(1)
    plt.ylabel('States')
    plt.xlabel('Actions')
    title = 'Policy Average for Player ' + str(agent_id + 1)
    if len(title_ads) > 0:
        for title_addition in title_ads:
            title = title + ' ' + title_addition
    ax.set_title(title, fontsize=title_font_size)
    plt.tight_layout()
    plt.close()
    if save:
        file_names = data_management.data_final_name(
            {'data_type': 'plot',
             'plot_type': 'policy_histogram',
             'domain_type': '1d',
             'num_agents': num_agents,
             'section': paper_figure['section'],
             'figure_id': paper_figure.get('figure_id', 'policy_histogram')},
            name_ads=name_ads,
            save_types=save_types,
            paper_figure=paper_figure['paper']
        )
        for file_name in file_names:
            fig.savefig(file_name, bbox_inches='tight')
    if axis_return:
        return ax
    else:
        return fig


def policy_deterministically_to_actions(env: influencer_env_async,
                                        q_table: dict = None,
                                        q_tensor: torch.Tensor = None,
                                        initial_position: np.ndarray = np.array([0, 1]),
                                        num_step: int = 10,
                                        temperature: float = 1,
                                        ) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Simulates deterministic actions for agents based on their policies. 
    By doing the following 
    
    1. The Q-table is converted to a policy using a softmax function.
    i.e.

    .. math::
        P(a|s) = \frac{e^{Q(s,a)/T}}{\sum_{a'} e^{Q(s,a')/T}}
    
    where:
        - :math:`a` is the action 
        - :math:`s` is the current state
        - :math:`a'` is the next state 
        - :math:`T` is the temperature parameter
        - :math:`P(a|s)` is the probability of taking action :math:`a` in state :math:`s`
        - :math:`Q(s,a)` is the Q-value for action :math:`a` in state :math:`s`
    
    2. The maximum action is selected for each state.
    3. The environment is stepped through the selected actions for a specified number of steps.
    4. The positions and rewards are recorded at each step.


    :param env: The environment object.
    :type env: influencer_env_async
    :param q_table: Q-table in dictionary format. Defaults to None.
    :type q_table: dict, optional
    :param q_tensor: Q-table as a torch.Tensor. Defaults to None.
    :type q_tensor: torch.Tensor, optional
    :param initial_position: Initial position of players. Defaults to np.array([0, 1]).
    :type initial_position: np.ndarray
    :param num_step: Number of steps to simulate. Defaults to 10.
    :type num_step: int
    :param temperature: A smoothness factor for the softmax function. Defaults to 1.
    :type temperature: float
    :return: Position matrix and reward matrix as torch.Tensors.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    if q_table is not None:
        q_tensor = IQL_utils.Q_table_to_tensor(q_table)
    env.initial_position = initial_position
    observations = env.reset()[0]
    agent_id = 0
    policies = []
    for agent in env.possible_agents:
        policy_agent = IQL_utils.Q_tensor_to_policy(q_tensor=q_tensor, temperature=temperature, agent_id=agent_id)
        max_policy = torch.max(policy_agent, dim=1)[1]
        policies.append(max_policy)
        agent_id += 1
    pos_matrix = []
    reward_matrix = []
    for step in range(num_step):
        action_dict = {}
        agent_id = 0
        for agent in env.possible_agents:
            max_policy = policies[agent_id]
            position_id = observations[agent]
            action = max_policy[position_id]
            action_dict[agent] = action
            agent_id += 1
        observations, rewards, _, _, _ = env.step(action_dict)

        reward_vec = torch.tensor([rewards[agent] for agent in env.possible_agents])
        pos_vec = torch.tensor(env.observation_to_position(observations))

        pos_matrix = general.matrix_builder(row_id=step, row=pos_vec, matrix=pos_matrix)
        reward_matrix = general.matrix_builder(row_id=step, row=reward_vec, matrix=reward_matrix)
    return pos_matrix, reward_matrix


def reward_plot(reward_matrix: torch.Tensor,
                possible_agents: dict,
                title_ads: List[str] = [],
                name_ads: List[str] = [],
                save: bool = False,
                save_types: List[str] = ['.png', '.svg'],
                font: dict = {'default_size': 12, 'title_size': 14, 'legend_size': 12, 'font_family': 'sans-serif'},
                paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'reward_plot'},
                figsize: Tuple[float, float] = (6, 6),
                axis_return: bool = False,
                ) -> matplotlib.figure.Figure:
    r"""
    Plots the rewards for all players over time.

    :param reward_matrix: Matrix containing rewards for each player at each step.
    :type reward_matrix: torch.Tensor
    :param possible_agents: Dictionary of possible agents in the environment.
    :type possible_agents: dict
    :param title_ads: List of additional title components to append. Default is ``[]``.
    :type title_ads: List[str], optional
    :param name_ads: List of additional name components for file saving. Default is ``[]``.
    :type name_ads: List[str], optional
    :param save: If ``True``, saves the figure to file. Default is ``False``.
    :type save: bool, optional
    :param save_types: List of file extensions for saving. Default is ``['.png', '.svg']``.
    :type save_types: List[str], optional
    :param font: Font configuration dictionary with keys ``'default_size'``, ``'title_size'``,
        ``'legend_size'``, and ``'font_family'``.
    :type font: dict, optional
    :param paper_figure: Paper figure configuration with keys ``'paper'``, ``'section'``,
        and ``'figure_id'``.
    :type paper_figure: dict, optional
    :param figsize: Figure size as (width, height). Default is ``(6, 6)``.
    :type figsize: Tuple[float, float], optional
    :param axis_return: If ``True``, returns the axes object instead of figure. Default is ``False``.
    :type axis_return: bool, optional
    :return: A figure of the reward through time using the optimal policy (or Axes if ``axis_return=True``).
    :rtype: matplotlib.figure.Figure or matplotlib.axes.Axes
    """
    font_family = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)

    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font_family})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_box_aspect(1)
    agent_id = 0
    num_agents = 0
    for agent in possible_agents:
        ax.plot(reward_matrix[:, agent_id], label=f"{agent}")
        agent_id += 1
        num_agents += 1
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    title = "Player Reward"
    if len(title_ads) > 0:
        for title_addition in title_ads:
            title = title + ' ' + title_addition
    ax.set_title(title, fontsize=title_font_size)
    ax.legend()
    plt.tight_layout()
    plt.close()
    if save:
        file_names = data_management.data_final_name(
            {'data_type': 'plot',
             'plot_type': 'MARL_reward',
             'domain_type': '1d',
             'num_agents': num_agents,
             'section': paper_figure['section'],
             'figure_id': paper_figure.get('figure_id', 'reward_plot')},
            name_ads=name_ads,
            save_types=save_types,
            paper_figure=paper_figure['paper']
        )
        for file_name in file_names:
            fig.savefig(file_name, bbox_inches='tight')
    if axis_return:
        return ax
    else:
        return fig


def pos_plot(pos_matrix: torch.Tensor,
             possible_agents: dict,
             domain_bounds: list,
             title_ads: List[str] = [],
             name_ads: List[str] = [],
             save: bool = False,
             save_types: List[str] = ['.png', '.svg'],
             font: dict = {'default_size': 12, 'title_size': 14, 'legend_size': 12, 'font_family': 'sans-serif'},
             paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'pos_plot'},
             figsize: Tuple[float, float] = (6, 6),
             axis_return: bool = False,
             ) -> matplotlib.figure.Figure:
    r"""
    Plots the positions of all players over time.

    :param pos_matrix: Matrix containing positions for each player at each step.
    :type pos_matrix: torch.Tensor
    :param possible_agents: Dictionary of possible agents in the environment.
    :type possible_agents: dict
    :param domain_bounds: List containing the lower and upper bounds of the domain.
    :type domain_bounds: list
    :param title_ads: List of additional title components to append. Default is ``[]``.
    :type title_ads: List[str], optional
    :param name_ads: List of additional name components for file saving. Default is ``[]``.
    :type name_ads: List[str], optional
    :param save: If ``True``, saves the figure to file. Default is ``False``.
    :type save: bool, optional
    :param save_types: List of file extensions for saving. Default is ``['.png', '.svg']``.
    :type save_types: List[str], optional
    :param font: Font configuration dictionary with keys ``'default_size'``, ``'title_size'``,
        ``'legend_size'``, and ``'font_family'``.
    :type font: dict, optional
    :param paper_figure: Paper figure configuration with keys ``'paper'``, ``'section'``,
        and ``'figure_id'``.
    :type paper_figure: dict, optional
    :param figsize: Figure size as (width, height). Default is ``(6, 6)``.
    :type figsize: Tuple[float, float], optional
    :param axis_return: If ``True``, returns the axes object instead of figure. Default is ``False``.
    :type axis_return: bool, optional
    :return: A figure of the agent positions through time using the optimal policy (or Axes if ``axis_return=True``).
    :rtype: matplotlib.figure.Figure or matplotlib.axes.Axes
    """
    font_family = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)

    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font_family})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_box_aspect(1)
    agent_id = 0
    num_agents = 0
    for agent in possible_agents:
        ax.plot(pos_matrix[:, agent_id], label=f"{agent}")
        agent_id += 1
        num_agents += 1
    ax.set_xlabel("Steps")
    ax.set_ylabel("Position")
    ax.set_ylim(domain_bounds[0], domain_bounds[1])
    title = "Player Position"
    if len(title_ads) > 0:
        for title_addition in title_ads:
            title = title + ' ' + title_addition
    ax.set_title(title, fontsize=title_font_size)
    ax.legend()
    plt.tight_layout()
    plt.close()
    if save:
        file_names = data_management.data_final_name(
            {'data_type': 'plot',
             'plot_type': 'MARL_pos',
             'domain_type': '1d',
             'num_agents': num_agents,
             'section': paper_figure['section'],
             'figure_id': paper_figure.get('figure_id', 'pos_plot')},
            name_ads=name_ads,
            save_types=save_types,
            paper_figure=paper_figure['paper']
        )
        for file_name in file_names:
            fig.savefig(file_name, bbox_inches='tight')
    if axis_return:
        return ax
    else:
        return fig


def agent_position_trajectory(pos_data: np.ndarray,
                               window_size: int = 10000,
                               num_agents: int = None,
                               title_ads: List[str] = [],
                               name_ads: List[str] = [],
                               short_title:bool=False,
                               save: bool = False,
                               save_types: List[str] = ['.png', '.svg'],
                               font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12, 'font_family': 'sans-serif'},
                               paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'agent_trajectory'},
                               figsize: Tuple[float, float] = (10, 6),
                               num_ticks: int = 6,
                               axis_return: bool = False,
                               ) -> matplotlib.figure.Figure:
    r"""
    Plots agent positions over time with scientific notation x-axis labels.
    
    This function visualizes the trajectory of agent positions throughout training,
    with x-axis labels showing the actual episode count (accounting for window averaging).
    The x-axis uses scientific notation for readability with large episode counts.

    Parameters
    ----------
    pos_data : np.ndarray
        Position data with shape ``(time_steps, num_agents)``. Assumes data is already 
        window-averaged (each row represents ``window_size`` episodes).
    window_size : int, optional
        The number of episodes averaged per data point (for x-axis labeling). Default is 10000.
    num_agents : int, optional
        Number of agents. If None, inferred from ``pos_data.shape[1]``.
    title_ads : List[str], optional
        List of additional title components to append. Default is ``[]``.
    name_ads : List[str], optional
        List of additional name components for file saving. Default is ``[]``.
    save : bool, optional
        If ``True``, saves the figure to file. Default is ``False``.
    save_types : List[str], optional
        List of file extensions for saving. Default is ``['.png', '.svg']``.
    font : dict, optional
        Font configuration dictionary with keys:
        
        - ``'default_size'``: Default font size (default: 12)
        - ``'title_size'``: Title font size (default: 14)
        - ``'legend_size'``: Legend font size (default: 12)
        - ``'font_family'``: Font family (default: 'sans-serif')
    paper_figure : dict, optional
        Paper figure configuration with keys:
        
        - ``'paper'``: bool, whether this is a paper figure
        - ``'section'``: str, section identifier
        - ``'figure_id'``: str, figure identifier
    figsize : Tuple[float, float], optional
        Figure size as (width, height). Default is ``(10, 6)``.
    num_ticks : int, optional
        Approximate number of x-axis ticks to display. Default is 6.
    axis_return : bool, optional
        If ``True``, returns the axes object instead of figure. Default is ``False``.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        Figure object (or Axes if ``axis_return=True``) showing agent position trajectories.

    Examples
    --------
    Basic usage with pre-averaged position data:

    >>> import numpy as np
    >>> pos_averaged = np.random.rand(400, 3)  # 400 windows, 3 agents
    >>> fig = agent_position_trajectory(pos_averaged, window_size=10000, 
    ...                                  title_ads=['Experiment 1'])

    Save figure for publication:

    >>> fig = agent_position_trajectory(pos_averaged, window_size=10000,
    ...                                  save=True, 
    ...                                  paper_figure={'paper': True, 'section': 'results', 'figure_id': 'trajectory'})
    """
    # Font setup
    font_family = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font_family})
    mpl.rcParams['legend.fontsize'] = legend_font_size
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_box_aspect(1)
    
    if num_agents is None:
        num_agents = pos_data.shape[1]
    num_points = len(pos_data)
    
    # Create x-values representing actual episode counts
    x_values = np.arange(1, num_points + 1) * window_size
    
    for i in range(num_agents):
        ax.plot(x_values, pos_data[:, i], label=f'Agent {i+1}')
    
    # Set x-ticks with scientific notation
    tick_spacing = max(1, num_points // num_ticks)
    tick_positions = np.arange(0, num_points + 1, tick_spacing) * window_size
    ax.set_xticks(tick_positions)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(default_font_size - 2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Position')
    
    # Build title
    if short_title:
        title="MARL Agents' Trajectories"
    else:
        title = f'{num_agents} MARL Agent Positions Over Training'
    if len(title_ads) > 0:
        for title_addition in title_ads:
            title = title + ' ' + title_addition
    ax.set_title(title, fontsize=title_font_size)
    
    ax.legend(loc='best')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.close()
    
    if save:
        file_names = data_management.data_final_name(
            {'data_type': 'plot', 
             'plot_type': 'MARL_trajectory', 
             'domain_type': '1d', 
             'num_agents': num_agents, 
             'section': paper_figure['section'], 
             'figure_id': paper_figure.get('figure_id', 'agent_trajectory')},
            name_ads=name_ads, 
            save_types=save_types, 
            paper_figure=paper_figure['paper']
        )
        for file_name in file_names:
            fig.savefig(file_name, bbox_inches='tight')
    
    if axis_return:
        return ax
    else:
        return fig

#

def bifurcation_over_parameters(positions: torch.Tensor,
                                 reach_parameters: np.ndarray,
                                 num_agents: np.ndarray,
                                 bin_points: np.ndarray,
                                 resource_distribution: np.ndarray,
                                 refinements: int = 10,
                                 plot_type: str = "heat",
                                 infl_cshift: bool = False,
                                 infl_type: str='gaussian',
                                 title_ads: List[str] = [],
                                 short_title: bool = False,
                                 name_ads: List[str] = [],
                                 save: bool = False,
                                 save_types: List[str] = ['.png', '.svg'],
                                 cmaps: dict = {'heat': 'Blues', 'trajectory': '#851321', 'crit': 'Greys'},
                                 font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'},
                                 cbar_config: dict = {'center_labels': True, 'label_alignment': 'center', 'shrink': 0.8},
                                 paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'equilibrium_bifurcation_plot'},
                                 axis_return: bool = False,
                                 show_pred: bool = False,
                                 marker_size: int = 30,
                                 ) -> None:
    heat_cmap = cmaps['heat']
    crit_cmap = cmaps['crit']
    font['font.family'] = cmaps.get('font_family', 'sans-serif')
    cbar_font_size= font.get('cbar_size', 12)
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    cbar_center_labels = cbar_config.get('center_labels', True)
    cbar_label_alignment = cbar_config.get('label_alignment', 'center')
    cbar_shrink = cbar_config.get('shrink', 1)

    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    reach_start = reach_parameters[0]
    reach_end = reach_parameters[-1]

    # Make sure inputs are torch tensors
    if not torch.is_tensor(reach_parameters):
        reach_parameters = torch.tensor(reach_parameters, dtype=torch.float32)
    domain = reach_parameters
    if not torch.is_tensor(bin_points):
        bin_points = torch.tensor(bin_points, dtype=torch.float32)
    if not torch.is_tensor(resource_distribution):
        resource_distribution = torch.tensor(resource_distribution, dtype=torch.float32)
    if not torch.is_tensor(positions):
        positions = torch.tensor(positions, dtype=torch.float32)

    # Create density heatmap showing number of agents at each position
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_box_aspect(1)

    # Scatter final positions per agent per sigma — compact SVG (mirrors AD approach)
    positions_np = positions.numpy() if torch.is_tensor(positions) else np.array(positions, dtype=float)
    reach_param_np = domain.numpy() if torch.is_tensor(domain) else np.array(domain, dtype=float)

    # Coarse density matrix for colormap scaling only (not rendered as image)
    _n_bins = 50
    _pos_bins = np.linspace(0, 1, _n_bins + 1)
    density_matrix = np.zeros((_n_bins, len(domain)))
    for i in range(min(len(domain), positions_np.shape[0])):
        valid = positions_np[i, ~np.isnan(positions_np[i, :])]
        if len(valid) > 0:
            counts, _ = np.histogram(valid, bins=_pos_bins)
            density_matrix[:, i] = counts

    max_agents = int(density_matrix.max()) if density_matrix.max() > 0 else num_agents
    levels = np.arange(0, max_agents + 2, 1)
    norm = mpl.colors.BoundaryNorm(levels, ncolors=256)

    # Gather all scatter coordinates with density-based color
    scatter_x, scatter_y, scatter_c = [], [], []
    for i in range(min(len(reach_param_np), positions_np.shape[0])):
        agent_positions_i = positions_np[i, :]
        valid_pos = agent_positions_i[~np.isnan(agent_positions_i)]
        for pos in valid_pos:
            bin_idx = int(np.clip(np.searchsorted(_pos_bins, pos, side='right') - 1, 0, _n_bins - 1))
            scatter_x.append(float(reach_param_np[i]))
            scatter_y.append(float(pos))
            scatter_c.append(float(density_matrix[bin_idx, i]))

    if len(scatter_x) > 0:
        im = ax.scatter(scatter_x, scatter_y, c=scatter_c,
                        cmap=heat_cmap, norm=norm, s=marker_size, linewidths=0, alpha=0.9)
    else:
        im = mpl.cm.ScalarMappable(cmap=heat_cmap, norm=norm)
        im.set_array([])

    # Keep reach_parameters in sync with domain for any downstream use
    reach_parameters = reach_parameters[:len(domain)]

    


    #Bifurcations critical values (works for gaussian only)
    if infl_type=='gaussian' and show_pred==True:
        _,means,crit_stds=one_utils.critical_values_plot(num_agents=num_agents,bin_points=bin_points,resource_distribution=resource_distribution,axis=ax,reach_start=reach_start,reach_end=reach_end,refinements=refinements,crit_cs=crit_cmap)
        crit_stds=general.flatten_list(xss=crit_stds)
        crit_stds.sort()
        for std in crit_stds:
            if std < reach_start or std > reach_end:
                #remove the std from crit_stds if it is outside the reach range
                crit_stds.remove(std)
            
        std_ticks = [float(np.around(i,decimals=2)) for i in np.linspace(reach_end, reach_start, num=5)]
        crit_means=np.around(means,decimals=3)
        mean_ticks= [0,1]+list(crit_means)
        mean_ticks.sort()

        std_removed=np.setdiff1d(np.array(std_ticks),np.around(crit_stds,2))
        std_tick_vals=np.array(list(std_removed)+crit_stds)
        std_tick_vals.sort()
        crit_std_locs=[]
        for std_id in range(len(crit_stds)):
            crit_std_locs.append(int(np.where(std_tick_vals==crit_stds[std_id])[0][0]))
        std_tick_labels=list(std_tick_vals.copy())
        for std_loc_id in range(len(crit_stds)):
            if std_loc_id ==len(crit_stds)-1:
                std_tick_labels[int(crit_std_locs[std_loc_id])]=r'$t_*$' #+r'='+str(std_tick_vals[crit_std_locs[std_loc_id]]) 
            else:
                std_tick_labels[int(crit_std_locs[std_loc_id])]=r'$t_'+str(len(crit_stds)-std_loc_id-1)+r'$' #+r'='+str(std_tick_vals[crit_std_locs[std_loc_id]])
        if infl_cshift==False:
            ax.xaxis.set_ticks(std_tick_vals)
            ax.xaxis.set_ticklabels(std_tick_labels)
            
        ax.yaxis.set_ticks(crit_means)
        ax.yaxis.set_ticklabels(crit_means)
    
    #Plot features
    if plot_type == "heat":
        # Create discrete colorbar for integer agent counts
        max_agents = int(density_matrix.max()) if 'density_matrix' in locals() else num_agents
        tick_levels = np.arange(0, max_agents + 2, 1)  # 0, 1, 2, ..., max_agents
        
        # Use the same norm as the image if it exists
        if 'norm' in locals():
            cbar = plt.colorbar(im, ax=ax, shrink=cbar_shrink, ticks=tick_levels)
        else:
            cbar = plt.colorbar(im, ax=ax, shrink=cbar_shrink, ticks=tick_levels)
        
        cbar.set_label('Number of Agents', fontsize=cbar_font_size)
        
        # Apply custom colorbar adjustments if requested
        if cbar_center_labels and len(tick_levels) > 1:
            # Calculate centered positions between tick marks
            centered_positions = []
            for i in range(len(tick_levels) - 1):
                centered_positions.append((tick_levels[i] + tick_levels[i + 1]) / 2)
            
            # Set centered tick positions and labels
            if len(centered_positions) > 0:
                cbar.set_ticks(centered_positions)
                # Create labels for centered positions - each represents the actual agent count for that color band
                # The first band (0-1) gets label "0", second band (1-2) gets label "1", etc.
                centered_labels = [str(i) for i in range(len(centered_positions))]
                cbar.set_ticklabels(centered_labels)
        else:
            # Keep original tick behavior when not centering
            cbar.set_ticks(tick_levels)
            cbar.set_ticklabels([str(int(level)) for level in tick_levels])
        
        # Apply label alignment if specified
        if cbar_label_alignment == 'center':
            cbar.ax.tick_params(axis='y', which='major', pad=5)
            for label in cbar.ax.get_yticklabels():
                label.set_horizontalalignment('center')


        

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if show_pred==True:
        plt.legend(by_label.values(), by_label.keys(),loc='lower center')
    if short_title:
        title= 'MARL Agents'
    else:
        title=str(num_agents)+' MARL Agents\' Bifurcation of Equilibria'
    if len(title_ads)>0:
        for title_addition in title_ads:
            title=title+" "+title_addition 
    plt.title(title,fontsize=title_font_size)
    if infl_type=='gaussian':
        plt.xlabel(r"$\sigma$ (std)")
    else: 
        plt.xlabel(r"$\sigma$")
    plt.ylim(0,1)
    plt.ylabel("Agent Position")
    plt.tight_layout()
    plt.close()
    if save==True:
        file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'MARL_bif','domain_type':'1d','num_agents':num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','position_at_equalibirum_histogram')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
        for file_name in file_names:
            fig.savefig(file_name,bbox_inches='tight')
    if axis_return:
        return ax
    else:
        return fig