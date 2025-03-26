import numpy as np
import torch
import random


from influencer_games.utils.rl_utils.MARL_utilities import *
from influencer_games.utils.rl_utils.IQL_utilities import *


class IQL_async():
    def __init__(self,config=None):
            super().__init__()
            self.configs=config
            self.random_seed = config['random_seed']
            self.env = config['env']
            self.epsilon_configs = config['epsilon_configs']
            self.gamma = config['gamma']
            self.alpha = config['alpha']
            self.epochs = config['epochs']
            self.random_initialize = config['random_initialize']
            self.soft_max=config['soft_max']
            self.episode_configs=config['episode_configs']
            self.observations,_ = self.env.reset()
            self.Q_table_intiation()
            

    def Q_table_intiation(self):
        self.Q_table = {}
        if self.random_initialize:
            self.Q_table = {player:{i:{j:np.random.uniform(low=0.0, high=.3) for j in range(self.env.action_spaces[player].n)} for i in range(self.env.observation_spaces[player].n)} for player in self.env.possible_agents}
        else :
            self.Q_table = {player:{i:{j:0 for j in range(self.env.action_spaces[player].n)} for i in range(self.env.observation_spaces[player].n)} for player in self.env.possible_agents}

    def observation_intialized(self):
        self.env.reset()
        self.observations={player:self.env.observation_spaces[player].sample() for player in self.env.possible_agents}
    
    def action_choice(self,episode,player):
        epsilon_f=Adjusted_epsilon(configs=self.epsilon_configs,num_players=self.env.num_players,episode=episode,episodes=self.episodes)
        epsilon = np.random.rand()
        action_space=self.env.action_spaces[player]
        if epsilon < epsilon_f:
            action =action_space.sample()
        else:
            if self.soft_max:
                action = self.Q_soft_max_action(player)
            else:
                action = self.Q_max_action(player)
        self.action_dict[player]=action
        return action

    
    
    def Q_soft_max_action(self,player):
        observation = self.observations[player]
        self.temparture=Adjusted_temperature(configs=self.configs["temperature_configs"],observation=observation,observation_space_size=self.env.observation_spaces[player].n)
        policy = np.array([np.exp(self.Q_table[player][observation][action] / self.temparture)  for action in range(self.env.action_spaces[player].n)])
        policy = policy/policy.sum()
        action = np.random.choice(range(self.env.action_spaces[player].n), p=policy)
        return action
        
    def Q_max_action(self,player):
        observation = self.observations[player]
        LEFT=0
        STAY=1
        RIGHT=2
        max_Q = max(self.Q_table[player][observation][LEFT],self.Q_table[player][observation][STAY],self.Q_table[player][observation][RIGHT])
        if max_Q == self.Q_table[player][observation][LEFT]:
            return LEFT
        elif max_Q == self.Q_table[player][observation][STAY]:
            return STAY
        else:
            return RIGHT
        
    def Q_step(self,episode):
        self.action_dict = {}
        agents=self.env.possible_agents
        random.shuffle(agents)
        for player in agents:
            action_dict = self.action_choice(episode,player)
            previous_observations = self.observations
            self.observations, self.rewards, terminateds, _ , _= self.env.step(self.action_dict,player)
            self.Q_table[player][previous_observations[player]][self.action_dict[player]] = (1-self.alpha) *self.Q_table[player][previous_observations[player]][self.action_dict[player]] + self.alpha*(self.rewards[player] + self.gamma * self.Q_table[player][self.observations[player]][self.Q_max_action(player)])
        return self.Q_table
    
    def train(self):
        for epoch in range(self.epochs):
            self.episodes=Adjusted_episodes(configs=self.episode_configs,epoch=epoch,epochs=self.epochs)
            for episode in range(self.episodes):
                self.Q_step(episode)
            self.observation_intialized()
                
        
        return self.Q_table
    