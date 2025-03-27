
"""
How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --sheldon-cooper-mode`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`

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
 


from influencer_games.domains.resource_distributions import *


bin_points=np.linspace(.001, .999, 100)




from influencer_games.rl_dynamics.influencer_game_sync import influencer_env_sync as influencer_env

from ray.rllib.connectors.env_to_module.flatten_observations import FlattenObservations
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,run_rllib_example_script_experiment,)
from ray.tune.registry import get_trainable_cls, register_env  # noqa



parser = add_rllib_example_script_args(
    default_reward=1, default_iters=50, default_timesteps=100000
)
parser.set_defaults(
    enable_new_api_stack=True,
    num_agents=2,
)

parser.add_argument(
    "--infl_type",
    type=str,
    default="gaussian",
    help="Influence type",
)



parser.add_argument(
    "--infl_parameter",
    type=float,
    default=.25,
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
    help="Fixed paraameter requred for the Dirlechet influence function", 
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

args_start = parser.parse_args()

parser.add_argument(
        "--intial_postion",
        type=np.ndarray,
        default=player_postion_setup(num_agents=args_start.num_agents,setup_type='intial_symmetric_setup',domain_type=args_start.domain_type,domain_bounds=args_start.domain_bounds),
        help="Inital positions of the agents",
    )

parser.add_argument(
    "--resource_distribution",
    type=np.ndarray,
    default=resource_distribution_choice(bin_points=args_start.bin_points,resource_type='multi_modal_gaussian_distribution_1D',resource_parameters=args_start.resource_parameters),
    help="Distribution of resources the players are competing for",
)

parser.add_argument(
    "--parameters",
    type=np.ndarray,
    default=player_parameter_setup(args_start.num_agents,infl_type=args_start.infl_type,setup_type="intial_symmetric_setup",reach=args_start.infl_parameter),
    help="Parameters of the influence function",
)
parser.add_argument(
    "--step_size",
    type=float,
    default=0.1,
    help="Player step size: determines how far the player can move in one step and size of the observation space",
)




if __name__ == "__main__":
    args = parser.parse_args()
    env_config_main={"num_agents": int(args.num_agents),
                "initial_postion":args.intial_postion,
                "bin_points":args.bin_points,
                "resource_distribution":args.resource_distribution,
                "step_size":args.step_size,
                "infl_type":args.infl_type,
                "parameters":args.parameters[0],
                "domain_type":args.domain_type,
                "domain_bounds":args.domain_bounds,
                "fixed_pa":args.fixed_pa,
                "NUM_ITERS":args.inside_iters,}

    # You can also register the env creator function explicitly with:
    register_env("my_env", lambda cfg: influencer_env(config=env_config_main))

    # Or you can hard code certain settings into the Env's constructor (`config`).
    # register_env(
    #    "rock-paper-scissors-w-sheldon-mode-activated",
    #    lambda config: RockPaperScissors({**config, **{"sheldon_cooper_mode": True}}),
    # )

    # Or allow the RLlib user to set more c'tor options via their algo config:
    # config.environment(env_config={[c'tor arg name]: [value]})
    # register_env("rock-paper-scissors", lambda cfg: RockPaperScissors(cfg))

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            "my_env",
        )
        .env_runners(
            env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
        )
        .multi_agent(
            # Define policies.
            policies={f"player{i}" for i in range(args.num_agents)},
            # Map agent "player1" to policy "player1" and agent "player2" to policy
            # "player2".
            policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
        )
    )


    result=run_rllib_example_script_experiment(base_config, args)