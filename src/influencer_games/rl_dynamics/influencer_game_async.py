import gymnasium as gym
from influencer_games.utils.utilities import *
from influencer_games.utils.rl_utils.MARL_utilities import prob_matrix

import gymnasium.spaces as spaces


from ray.rllib.env.multi_agent_env import MultiAgentEnv

class influencer_env_async(MultiAgentEnv):
    """Two-player environment for the famous rock paper scissors game.
   
    Both players always move simultaneously over a course of 10 timesteps in total.
    The winner of each timestep receives reward of +1, the losing player -1.0.

    The observation of each player is the last opponent action.
    """

    
    

    def __init__(self,config=None):
        super().__init__()
        self.num_players = config.get('num_agents')
        self.intial_postion=config.get('initial_postion')
        self.bin_points=config.get('bin_points')
        self.resource_distribution=config.get('resource_distribution')
        self.step_size=config.get('step_size')
        self.domain_type=config.get('domain_type')
        self.domain_bounds=config.get('domain_bounds')
        self.infl_configs=config.get('infl_configs')
        self.infl_type=self.infl_configs['infl_type']
        self.parameters=config.get('parameters')
        self.fixed_pa=config.get('fixed_pa')
        self.NUM_ITERS=config.get('NUM_ITERS')
        self.domain_points_num=int((self.domain_bounds[1]-self.domain_bounds[0])/self.step_size)+1
        self.possible_postions=np.linspace(self.domain_bounds[0],self.domain_bounds[1],num=self.domain_points_num)
        

        self.agents = self.possible_agents = [f"player{i}" for i in range(self.num_players)]
        # The observations are always the last taken actions. Hence observation- and
        # action spaces are identical.
        self.observation_spaces = { f"player{i}":gym.spaces.Discrete(self.domain_points_num) for i in range(self.num_players)}
        self.action_spaces = { f"player{i}":gym.spaces.Discrete(3) for i in range(self.num_players)}
        self.last_move = None
        self.num_moves = 0
        self.LEFT = 0
        self.STAY = 1
        self.RIGHT = 2

    def REWARD_MAP(self,observations)->int|torch.Tensor:
        postion=self.observation_to_postion(observations)
        postion=np.array(postion)
        pr_matrix=prob_matrix(num_players=self.num_players,agents_pos=postion,bin_points=self.bin_points,infl_configs=self.infl_configs,parameters=self.parameters,fixed_pa=self.fixed_pa)
        reward_vec=torch.sum(pr_matrix*torch.tensor(self.resource_distribution),1)
        reward={self.possible_agents[agent_id]: reward_vec[agent_id].item() for agent_id in range(self.num_players)}
        return reward

    def reset(self, *, seed=None, options=None):
        self.num_moves = 0

        # The first observation should not matter (none of the agents has moved yet).
        # Set them to 0.
        observations_list=self.intial_postion_to_observation()
        observations = {self.possible_agents[agent_id]: observations_list[agent_id] for agent_id in range(self.num_players)}
        self.observations=observations
        return observations, {}  # <- empty infos dict
    
    def intial_postion_to_observation(self):
        observations_list=[]
        for pos in self.intial_postion:
            idx=int(np.where(np.around(self.possible_postions,decimals=5)==pos)[0][0])
            observations_list.append(idx)
            
        return observations_list
    
    def observation_to_postion(self,observations):
        pos_list=[]
        for key in observations:
            position=self.possible_postions[int(observations[key])]
            pos_list.append(position)
        return pos_list
    
    def observation_update(self,actions,observations,key):
        if self.domain_type=='1d':
            if actions[key]==self.LEFT:
                new_obs=int(int(observations[key])-1)
            elif actions[key]==self.STAY:
                new_obs=int(observations[key])
            elif actions[key]==self.RIGHT:
                new_obs=int(int(observations[key])+1)
            if new_obs<0 or new_obs>self.domain_points_num-1:
                new_obs=int(observations[key])
            else:
                observations[key]=new_obs

        return observations
    
    def step(self, action_dict, player):
        self.num_moves += 1/self.num_players

        # Set the next observations (simply use the other player's action).
        # Note that because we are publishing both players in the observations dict,
        # we expect both players to act in the next `step()` (simultaneous stepping).
        observations = self.observation_update(actions=action_dict,observations=self.observations,key=player)

        # Compute rewards for each player based on the win-matrix.
        rewards = self.REWARD_MAP(observations)

        # Terminate the entire episode (for all agents) once 10 moves have been made.
        terminateds = {"__all__": self.num_moves >= self.NUM_ITERS}

        # Leave truncateds and infos empty.
        return observations, rewards, terminateds, {}, {}

