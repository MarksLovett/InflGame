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

def add_rl_example_script_args(
    parser: Optional[argparse.ArgumentParser] = None,
    default_reward: float = 100.0,
    default_iters: int = 200,
    default_timesteps: int = 100000,
) -> argparse.ArgumentParser:
    """Adds RLlib-typical (and common) examples scripts command line args to a parser.

    TODO (sven): This function should be used by most of our examples scripts, which
     already mostly have this logic in them (but written out).

    Args:
        parser: The parser to add the arguments to. If None, create a new one.
        default_reward: The default value for the --stop-reward option.
        default_iters: The default value for the --stop-iters option.
        default_timesteps: The default value for the --stop-timesteps option.

    Returns:
        The altered (or newly created) parser object.
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


    #Enviroment parameters
    parser.add_argument(
        "--num_agents",
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
        default={'infl_type':'gaussian'},
        help="Influence configs",
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
        "--resource_type",
        type=str,
        default="multi_modal_gaussian_distribution_1D",
        help="Type of reward distribution"
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

    parser.add_argument(
        "--step_size",
        type=float,
        default=0.1,
        help="Player step size: determines how far the player can move in one step and size of the observation space",
    )

    
    # Second level args, from user input
    
    args_pre = parser.parse_args(args=[])

    parser.add_argument(
        "--intial_postion",
        type=np.ndarray,
        default=player_postion_setup(num_agents=3,setup_type='intial_symmetric_setup',domain_type=args_pre.domain_type,domain_bounds=args_pre.domain_bounds),
        help="Inital positions of the agents",
    )

    parser.add_argument(
        "--resource_distribution",
        type=np.ndarray,
        default=resource_distribution_choice(bin_points=args_pre.bin_points,resource_type=args_pre.resource_type,resource_parameters=args_pre.resource_parameters),
        help="Distribution of resources the players are competing for",
    )

    parser.add_argument(
        "--parameters",
        type=np.ndarray,
        default=player_parameter_setup(num_agents=args_pre.num_agents,infl_type=args_pre.infl_type,setup_type="intial_symmetric_setup",reach=args_pre.infl_parameter),
        help="Parameters of the influence function",
    )
    
    args_env = parser.parse_args(args=[])

    parser.add_argument(
        "--env_configs",
        type=np.ndarray,
        default={"num_agents": args_env.num_agents,"initial_postion":args_env,"bin_points":args_env.bin_points,"resource_distribution":args_env.resource_distribution,"step_size":args_env.step_size,"infl_type":args_env.infl_type,"parameters":args_env.parameters,"domain_type":args_env.domain_type,"domain_bounds":args_env.domain_bounds,"fixed_pa":args_env.fixed_pa,"NUM_ITERS":args_env.inside_iters,}
        help="Parameters of the influence function",
    )


    #Algorithm parameters

    parser.add_argument(
        "--gamma",
        type=float,
        default=10,
        help="gamma (discount factor) value",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5000,
        help="The number of traning ephocs the alogrithm is run",
    )

    ##episode configs
    
    parser.add_argument(
        "--episode_type",
        type=str,
        default="Cosine_Anneling",
        help="How the episode length changes",
    )

    parser.add_argument(
        "--episode_max",
        type=int,
        default=100,
        help="Longest amount of episodes trained through",
    )

    parser.add_argument(
        "--episode_min",
        type=int,
        default=10,
        help="Shortest amount of episodes trained through",
    )

    ##temperature (smoothing) configs

    parser.add_argument(
        "--temperature_type",
        type=str,
        default="Cosine_Anneling_Distance",
        help="How the tempature strength changes",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="fixed temperature",
    )

    parser.add_argument(
        "--temperature_max",
        type=float,
        default=1,
        help="Largest global temperature and/or of the first segment",
    )

    parser.add_argument(
        "--temperature_local_min",
        type=float,
        default=0.7,
        help="Smallest local temperature of the first segment",
    )

    parser.add_argument(
        "--temperature_local_max",
        type=float,
        default=0.7,
        help="Largest local temperature of the second segment",
    )
    parser.add_argument(
        "--temperature_min",
        type=float,
        default=0.5,
        help="Global minimum temperature and/or of the second segment",
    )


    ##epsilon (exploration rate) configs

    parser.add_argument(
        "--episilon_type",
        type=str,
        default="Cosine_Anneling",
        help="How epsilon changes",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=.3,
        help="Fixed epsilon rate",
    )

    parser.add_argument(
        "--episilon_max",
        type=float,
        default=.8,
        help="Largest exploration rate",
    )

    parser.add_argument(
        "--epislon_min",
        type=float,
        default=.3,
        help="Smallest epsilon rate",
    )

   

    ##configs dicts
    args_start = parser.parse_args(args=[])
    ### episodes
    if args_start.episode_type=="Normal":
        parser.add_argument(
            "--episode_configs",
            type=dict,
            default={"TYPE":args_start.episode_type,"episode_max":args_start.episode_max},
            help="defualt configs",
        )
    elif args_start.episode_type=="Reverse_Cosine_Anneling":
        parser.add_argument(
            "--episode_configs",
            type=dict,
            default={"TYPE":args_start.episode_type,"episode_max":args_start.episode_max,"episode_min":args_start.episode_min},
            help="defualt configs",
        )
    
    ###temperature
    if args_start.temperature_type=="Normal":
        parser.add_argument(
            "--temp_configs",
            type=dict,
            default={"TYPE":args_start.temperature_type,"temperature":args_start.temperature},
            help="defualt configs",
        )
    elif args_start.temperature_type=="Cosine_Anneling_Distance":
        parser.add_argument(
            "--temp_configs",
            type=dict,
            default={"TYPE":args_start.temperature_type,"temperature_max":args_start.temperature_local_min,"temperature_min":args_start.temperature_min},
            help="defualt configs",
        )
    elif args_start.temperature_type=="Cosine_Anneling_Distance_segmented":
        parser.add_argument(
            "--temp_configs",
            type=dict,
            default={"TYPE":args_start.temperature_type,"temperature_max":args_start.temperature_max,"temperature_local_min":args_start.temperature_local_min,"temperature_local_max":args_start.temperature_local_max,"temperature_min":args_start.temperature_min},
            help="defualt configs",
        )

    ###epsilon

    if args_start.episilon_type=="Normal":
        parser.add_argument(
            "--episilon_configs",
            type=dict,
            default={"TYPE":args_start.episilon_type,'epsilon':args_start.epsilon},
            help="defualt configs",
        )

    elif args_start.episilon_type=="Cosine_Anneling":
        parser.add_argument(
            "--episilon_configs",
            type=dict,
            default={"TYPE":args_start.episilon_type,"e_max":args_start.episilon_max,"e_min":args_start.epislon_min},
            help="defualt configs",
        )
    

    
    


    



    return parser