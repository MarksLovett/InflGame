"""
.. module:: my_parse
   :synopsis: Provides argument parsing utilities for RLlib-based scripts in the influencer games framework.

RLlib Argument Parsing Utilities Module
=======================================

This module provides utility functions for parsing command-line arguments for RLlib-based 
scripts. It is adapted from the RAYRL library to include custom configurations and parameters 
specific to the influencer games project.

The modifications are tailored to support multi-agent reinforcement learning experiments 
in the context of influencer games, with additional arguments 
and configurations unique to this domain.

Features:
---------
- Adds RLlib-typical example script arguments.
- Supports custom configurations for multi-agent reinforcement learning experiments.
- Includes domain-specific parameters for resource distribution and influence modeling.

Dependencies:
-------------
- InflGame.utils
- InflGame.domains

Usage:
------
The `add_rl_example_script_args` function adds RLlib-typical example script arguments to a parser, 
allowing for customization of reinforcement learning experiments in the influencer games framework.

Example:
--------

.. code-block:: python

    import argparse
    from InflGame.MARL.utils.my_parse import add_rl_example_script_args

    # Create a parser and add RLlib arguments
    parser = argparse.ArgumentParser()
    parser = add_rl_example_script_args(parser)

    # Parse arguments
    args = parser.parse_args()

    # Access parsed arguments
    print("Algorithm:", args.algo)
    print("Number of agents:", args.num_agents)
    print("Domain type:", args.domain_type)
    print("Initial positions:", args.initial_position)

"""

import argparse
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
import numpy as np
import InflGame.utils.general as general
import InflGame.domains.rd as rd

def add_rl_example_script_args(
    parser: Optional[argparse.ArgumentParser] = None,
    default_reward: float = 100.0,
    default_iters: int = 200,
    default_timesteps: int = 100000,
) -> argparse.ArgumentParser:
    """
    Adds RLlib-typical (and common) example script command line arguments to a parser.

    .. note::
        This function should be used by most of our example scripts, which
        already mostly have this logic in them (but written out).

    :param parser: The parser to add the arguments to. If None, create a new one.
    :type parser: Optional[argparse.ArgumentParser]
    :param default_reward: The default value for the ``--stop-reward`` option.
    :type default_reward: float
    :param default_iters: The default value for the ``--stop-iters`` option.
    :type default_iters: int
    :param default_timesteps: The default value for the ``--stop-timesteps`` option.
    :type default_timesteps: int

    :return: The altered (or newly created) parser object.
    :rtype: argparse.ArgumentParser
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    # Algo and Algo config options.
    parser.add_argument(
        "--algo", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument(
        "--enable-new-api-stack",
        action="store_true",
        help="Whether to use the `enable_rl_module_and_learner` config setting.",
    )
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="The gym.Env identifier to run the experiment with.",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=None,
        help="The number of (remote) EnvRunners to use for the experiment.",
    )
    parser.add_argument(
        "--num-envs-per-env-runner",
        type=int,
        default=None,
        help="The number of (vectorized) environments per EnvRunner. Note that "
        "this is identical to the batch size for (inference) action computations.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="If 0 (default), will run as single-agent. If > 0, will run as "
        "multi-agent with the environment simply cloned n times and each agent acting "
        "independently at every single timestep. The overall reward for this "
        "experiment is then the sum over all individual agents' rewards.",
    )
    

    parser.add_argument(
        "--infl_configs",
        type=dict,
        default={"infl_type":"gaussian"},
        help="Influence type",
    )

    parser.add_argument(
        "--infl_parameter",
        type=float,
        default=.35,
        help="reach distance",
    )
    parser.add_argument(
        "--bin_points",
        type=np.ndarray,
        default=np.linspace(.001, .999, 100),
        help="Positions of rewards",
    )

    parser.add_argument(
        "--resource_parameters",
        type=np.ndarray,
        default=[[.1,.1],[.25,.75],[1,1]],
        help="parameters of the reward distribution"
    )

    parser.add_argument(
        "--domain_type",
        type=str,
        default="1d",
        help="Type of domain for the game",
    )

    parser.add_argument(
        "--fixed_pa",
        type=int,
        default=0,
        help="Fixed parameter required for the Dirichlet influence function", 
    )

    parser.add_argument(
        "--domain_bounds",
        type=np.ndarray,
        default=np.array([0,1]),
        help="Bounds of the domain",
    )

    parser.add_argument(
        "--inside_iters",
        type=int,
        default=10,
        help="NUM_ITERS before the game is terminated",
    )

    args_start = parser.parse_args(args=[])

    parser.add_argument(
            "--initial_position",
            type=np.ndarray,
            default=general.agent_position_setup(num_agents=3,setup_type='initial_symmetric_setup',domain_type=args_start.domain_type,domain_bounds=args_start.domain_bounds),
            help="Initial positions of the agents",
        )

    parser.add_argument(
        "--resource_distribution",
        type=np.ndarray,
        default=rd.resource_distribution_choice(bin_points=args_start.bin_points,resource_type='multi_modal_gaussian_distribution_1D',resource_parameters=args_start.resource_parameters),
        help="Distribution of resources the players are competing for",
    )

    parser.add_argument(
        "--parameters",
        type=np.ndarray,
        default=general.agent_parameter_setup(3,infl_type=args_start.infl_configs["infl_type"],setup_type="initial_symmetric_setup",reach=args_start.infl_parameter),
        help="Parameters of the influence function",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.1,
        help="Player step size: determines how far the player can move in one step and size of the observation space",
    )




    # Evaluation options.
    parser.add_argument(
        "--evaluation-num-env-runners",
        type=int,
        default=0,
        help="The number of evaluation (remote) EnvRunners to use for the experiment.",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=0,
        help="How many iterations to run one round of evaluation. "
        "Use 0 (default) to disable evaluation.",
    )
    parser.add_argument(
        "--evaluation-duration",
        type=lambda v: v if v == "auto" else int(v),
        default=10,
        help="The number of evaluation units to run each evaluation round. "
        "Use `--evaluation-duration-unit` to count either in 'episodes' "
        "or 'timesteps'. If 'auto', will run as many as possible during train pass ("
        "`--evaluation-parallel-to-training` must be set then).",
    )
    parser.add_argument(
        "--evaluation-duration-unit",
        type=str,
        default="episodes",
        choices=["episodes", "timesteps"],
        help="The evaluation duration unit to count by. One of 'episodes' or "
        "'timesteps'. This unit will be run `--evaluation-duration` times in each "
        "evaluation round. If `--evaluation-duration=auto`, this setting does not "
        "matter.",
    )
    parser.add_argument(
        "--evaluation-parallel-to-training",
        action="store_true",
        help="Whether to run evaluation parallel to training. This might help speed up "
        "your overall iteration time. Be aware that when using this option, your "
        "reported evaluation results are referring to one iteration before the current "
        "one.",
    )

    # RLlib logging options.
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="The output directory to write trajectories to, which are collected by "
        "the algo's EnvRunners.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,  # None -> use default
        choices=["INFO", "DEBUG", "WARN", "ERROR"],
        help="The log-level to be used by the RLlib logger.",
    )

    # tune.Tuner options.
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Whether to NOT use tune.Tuner(), but rather a simple for-loop calling "
        "`algo.train()` repeatedly until one of the stop criteria is met.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="How many (tune.Tuner.fit()) experiments to execute - if possible in "
        "parallel.",
    )
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=None,
        help="How many (tune.Tuner) trials to run concurrently.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="The verbosity level for the `tune.Tuner()` running the experiment.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help=(
            "The frequency (in training iterations) with which to create checkpoints. "
            "Note that if --wandb-key is provided, all checkpoints will "
            "automatically be uploaded to WandB."
        ),
    )
    parser.add_argument(
        "--checkpoint-at-end",
        action="store_true",
        help=(
            "Whether to create a checkpoint at the very end of the experiment. "
            "Note that if --wandb-key is provided, all checkpoints will "
            "automatically be uploaded to WandB."
        ),
    )

    # WandB logging options.
    parser.add_argument(
        "--wandb-key",
        type=str,
        default=None,
        help="The WandB API key to use for uploading results.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="The WandB project name to use.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="The WandB run name to use.",
    )

    # Experiment stopping and testing criteria.
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=default_reward,
        help="Reward at which the script should stop training.",
    )
    parser.add_argument(
        "--stop-iters",
        type=int,
        default=default_iters,
        help="The number of iterations to train.",
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=default_timesteps,
        help="The number of (environment sampling) timesteps to train.",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test. If set, --stop-reward must "
        "be achieved within --stop-timesteps AND --stop-iters, otherwise this "
        "script will throw an exception at the end.",
    )
    parser.add_argument(
        "--as-release-test",
        action="store_true",
        help="Whether this script should be run as a release test. If set, "
        "all that applies to the --as-test option is true, plus, a short JSON summary "
        "will be written into a results file whose location is given by the ENV "
        "variable `TEST_OUTPUT_JSON`.",
    )

    # Learner scaling options.
    parser.add_argument(
        "--num-learners",
        type=int,
        default=None,
        help="The number of Learners to use. If `None`, use the algorithm's default "
        "value.",
    )
    parser.add_argument(
        "--num-cpus-per-learner",
        type=float,
        default=None,
        help="The number of CPUs per Learner to use. If `None`, use the algorithm's "
        "default value.",
    )
    parser.add_argument(
        "--num-gpus-per-learner",
        type=float,
        default=None,
        help="The number of GPUs per Learner to use. If `None` and there are enough "
        "GPUs for all required Learners (--num-learners), use a value of 1, "
        "otherwise 0.",
    )
    parser.add_argument(
        "--num-aggregator-actors-per-learner",
        type=int,
        default=None,
        help="The number of Aggregator actors to use per Learner. If `None`, use the "
        "algorithm's default value.",
    )

    # Ray init options.
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    # Old API stack: config.num_gpus.
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="The number of GPUs to use (only on the old API stack).",
    )

    return parser