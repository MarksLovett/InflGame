import numpy as np
import torch
from influencer_games.utils.utilities import *
from influencer_games.infl_kernels.gaussian_influence import *
from influencer_games.infl_kernels.Jones_influence import *
from influencer_games.infl_kernels.dirl_influence import *
from influencer_games.infl_kernels.MVG_influence import *
from influencer_games.domains.simplex.simplex_utlities import *

class AdapativeEnv:
    def __init__(self,
                 num_agents:int,
                 agents_pos:list|np.ndarray,
                 parameters:torch.Tensor,
                 resource_distribution:torch.Tensor,
                 bin_points:list|np.ndarray,
                 infl_configs:dict = {'infl_type':'gaussian'},
                 lr_type:str = 'cosine',
                 learning_rate:list = [.0001,.01,15],
                 time_steps:int = 100,
                 fp:int = 0,
                 infl_cshift:bool = False,
                 cshift:int = 0,
                 infl_fshift:bool = False,
                 Q:int = 0,
                 domain_type:str = '1d',
                 domain_bounds:list[float]|torch.Tensor = [0,1],
                 tolarance:float = 10**-5,
                 tolarated_agents:int = None, 
                 ):
        
        self.num_agents=num_agents
        self.agents_pos = agents_pos
        self.infl_type=infl_configs['infl_type']
        self.infl_configs=infl_configs
        self.parameters=parameters
        self.resource_distribution=resource_distribution
        self.bin_points=bin_points
        self.learning_rate=learning_rate
        self.time_steps=time_steps
        self.fixed_pa=fp
        self.lr_type=lr_type
        self.infl_cshift=infl_cshift
        self.cshift=cshift
        self.infl_fshift=infl_fshift
        self.Q=Q
        self.domain_type=domain_type
        self.domain_bounds=domain_bounds
        self.sigma_inv=0
        self.tolarance=tolarance
        if tolarated_agents == None:
            tolarated_agents = num_agents
        else:
            self.tolarated_agents=tolarated_agents
        

    def influence(self,
                  agent_id:int,
                  parameter_instance:list|np.ndarray|torch.Tensor,
                  alpha_matrix:torch.Tensor=0,
                  )->torch.Tensor:
        """
        Takes the agent's postion and parameters and finds the influence wrt to the method you have given
        INPUTS:
            agent_id:The current player\agent's id
            parameter_instance: parameter(s) unique to your influence distribution
            alpha_matrix:Unique to Dirlechet influence, alpha parmaters 
        OUTPUTS:
            infl: influence vetor for current agent ID, each value corresponds to influence over the bin/resource-point at that location
        """

        if self.infl_cshift==True and agent_id==self.num_agents:
            infl=torch.tensor(self.cshift)
        elif self.infl_fshift==True and agent_id>=self.num_agents:
            #This part determines if we are shifting our influence matrix by a custom function (right now just takes the abstaining function)
            infl=[]
            if len(self.agents_pos.shape)>1:
                print('Not done yet')
            else:
                for bin_point in self.bin_points:
                   infl_instance=1
                   for pos in self.agents_pos:
                        infl_instance=infl_instance*(bin_point-pos)**2
                   infl_instance=self.Q*infl_instance
                   infl.append(infl_instance)
                infl=torch.tensor(infl)
        else:
            if self.infl_type=='gaussian':
                infl=gaussian_infl(agent_id=agent_id,parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)
            
            elif self.infl_type=='Jones_M':
                infl=jones_infl(agent_id=agent_id,parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)

            elif self.infl_type=='dirl':
                infl=dirl_infl(agent_id=agent_id,bin_points=self.bin_points,alpha_matrix=alpha_matrix)

            elif self.infl_type=='multi_gaussian':
                self.sigma_inv=MVG_cov_matrix(parameter_instance=parameter_instance)

                infl=MVG_infl(agent_id=agent_id,agents_pos=self.agents_pos,bin_points=self.bin_points,sigma_inv=self.sigma_inv)
            elif self.infl_type=='custom':
                custom_influence=self.infl_configs['custom_influence']
                x_torch=torch.tensor(self.agents_pos[agent_id])
                p=np.array([parameter_instance[agent_id]])
                infl=custom_influence(x_torch,bin_points=self.bin_points,parameter_instance=p[0])
            else:
                print('no method selected!') 
        return infl
    
    def influence_matrix(self,
                         parameter_instance:list|np.ndarray|torch.Tensor,
                         )->torch.Tensor:
        # Takes the agent's postion and parameters and uses the influence function to make a matrix
        # INPUTS:
        #   agent_id:The current player\agent's id
        #   parameter_instance: parameter(s) unique to your influence distribution
        # OUTPUTS: 
        #   infl_matrix: Matrix of agnets' influences over every bin_point/resource_point

        
        #For Dirlechet influence only
        if self.infl_type=='dirl':
            alpha_matrix=dirl_parm(num_agents=self.num_agents,parameter_instance=parameter_instance,agents_pos=self.agents_pos,fixed_pa=self.fixed_pa)
        else:
            alpha_matrix=0
        self.alpha_matrix=alpha_matrix

        #For types of shifts
        agents=self.num_agents
        if self.infl_cshift==True:
            agents=agents+1
        if self.infl_fshift==True:
            agents=agents+1
        
        #Assembling the influence matrix for all agents
        infl_matrix=0
        for agent_id in range(agents):
            infl_row=self.influence(agent_id,parameter_instance,alpha_matrix)

            infl_matrix=matrix_builder(row_id=agent_id,row=infl_row,matrix=infl_matrix)
            
        return infl_matrix
    

    def prob_matrix(self,
                    parameter_instance:list|np.ndarray|torch.Tensor,
                    ):
        # Takes the agents' influence matrix and makes a proability matrix with each row index corresponding to the agent's ID
        # INPUTS:
        #   parameter_instance: parameter(s) unique to your influence distribution
        # OUTPUTS:
        #   agent_prob_matrix:Matrix of proabilities of a player/agent influenceing each bin-point/resource-point

        infl_matrix=self.influence_matrix(parameter_instance)
        denom=torch.sum(infl_matrix, 0)
        numer=infl_matrix
        agent_prob_matrix=numer/denom
        
        return agent_prob_matrix
    
    def reward_F(self,
                 parameter_instance:list|np.ndarray|torch.Tensor,
                 )->int|torch.Tensor:
        # Takes the agents' prob_matrixability matrix and makes a reward value for all agents
        # INPUTS:
        #   parameter_instance: parameter(s) unique to your influence distribution
        # OUTPUTS:
        #   reward: Agents' current reward

        pr_matrix=self.prob_matrix(parameter_instance)
        reward=torch.sum(pr_matrix*torch.tensor(self.resource_distribution),1)
        return reward


    def d_lnf_matrix(self,
                     parameter_instance:list|np.ndarray|torch.Tensor,
                     )->int|torch.Tensor:
        # Takes the agents' infleunce matrix and finds the log derivative of the 
        # INPUTS:
        #   parameter_instance: parameter(s) unique to your influence distribution
        # OUTPUTS:
        #   reward: Agent's current reward
        d_matrix=0
        for agent_id in range(self.num_agents):
            if self.infl_type=='gaussian':
                d_row=d_gaussian(agent_id=agent_id,parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)

            elif self.infl_type=='Jones_M':
                d_row=d_Jones(agent_id=agent_id,parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)

            elif self.infl_type=='dirl':

                self.alpha_matrix=dirl_parm(num_agents=self.num_agents,parameter_instance=parameter_instance,agents_pos=self.agents_pos,fixed_pa=self.fixed_pa)

                d_row=d_dirl(agent_id,agents_pos=self.agents_pos,bin_points=self.bin_points,alpha_matrix=self.alpha_matrix,fixed_pa=self.fixed_pa)
            
            elif self.infl_type=='multi_gaussian':
                self.sigma_inv=MVG_cov_matrix(parameter_instance=parameter_instance)
                
                d_row=d_MVG(agent_id=agent_id,agents_pos=self.agents_pos,bin_points=self.bin_points,sigma_inv=self.sigma_inv)
                

            d_matrix=matrix_builder(agent_id,d_row,d_matrix)
        
        return d_matrix 
    

    def shift_matrix(self,
                    parameter_instance:list|np.ndarray|torch.Tensor,
                    )->torch.Tensor:
        # This function is only used if we want shift our matrix in 1-D, it represents abstaining voters in this case. 
        # The shift matrix is used for shifting the gradient
        # INPUTS:
        #   parameter_instance: parameter(s) unique to your influence distribution
        # OUTPUTS: 
        #   shift_matrix: Matrix of how the functional shift influences agents' gradient.

        infl_matrix=self.influence_matrix(parameter_instance)
        denom=torch.sum(infl_matrix, 0)
        shift_matrix=0
        for agent_id in range(self.num_agents):
            shift_row=[]
            if agent_id==0:
                for bin_point in self.bin_points:
                    shift_instance=1
                    for pos in self.agents_pos[1:]:
                        shift_instance=shift_instance*(bin_point-pos)**2
                    shift_instance=-2*self.Q*shift_instance*(bin_point-self.agents_pos[agent_id])
                    shift_row.append(shift_instance)
            elif agent_id==self.num_agents-1:
                for bin_point in self.bin_points:
                    shift_instance=1
                    for pos in self.agents_pos[:-1]:
                        shift_instance=shift_instance*(bin_point-pos)**2
                    shift_instance=-2*self.Q*shift_instance*(bin_point-self.agents_pos[agent_id])
                    shift_row.append(shift_instance)  
            else:
                for bin_point in self.bin_points: 
                    shift_instance=1
                    for pos in np.concatenate((self.agents_pos[:agent_id],self.agents_pos[agent_id+1:]), axis=0):
                        shift_instance=shift_instance*(bin_point-pos)**2
                    shift_instance=-2*self.Q*shift_instance*(bin_point-self.agents_pos[agent_id])
                    shift_row.append(shift_instance)
            shift_row=torch.tensor(shift_row)
            shift_matrix=matrix_builder(row_id=agent_id,row=shift_row,matrix=shift_matrix)

        shift_matrix=shift_matrix/denom
        return shift_matrix
    
    def d_torch(self,parmaeter_instance:list|np.ndarray|torch.Tensor):
        d_matrix=0
        if self.domain_type=='1d':
            for player_id in range(self.num_agents):
                x_torch=torch.tensor([self.agents_pos[player_id]]*len(self.bin_points),requires_grad=True)
                external_grad = torch.tensor([1.]*len(self.bin_points))
                custom_influence=self.infl_configs['custom_influence']
                infl_row=torch.log(custom_influence(x_torch,self.bin_points,parmaeter_instance[player_id]))
                infl_row.backward(gradient=external_grad)
                d_row=x_torch.grad
                d_matrix=matrix_builder(row_id=player_id,row=d_row,matrix=d_matrix)
        else:
            for player_id in range(self.num_agents):
                x_torch=torch.tensor(self.agents_pos[player_id])
                x=x_torch.repeat(len(self.bin_points),1)
                x.requires_grad=True
                external_grad=torch.tensor([1.]*len(self.bin_points))
                custom_influence=self.infl_configs['custom_influence']
                infl_row=torch.log(custom_influence(x,self.bin_points,parmaeter_instance[player_id]))
                infl_row.backward(gradient=external_grad)
                d_row=[]
                for dim in range(len(x_torch)):
                    d_row.append(x.grad[:,dim])
                d_row=torch.stack(d_row)
                d_matrix=matrix_builder(row_id=player_id,row=d_row,matrix=d_matrix)
                
        return d_matrix



    def gradient(self,
                 parameter_instance:list|np.ndarray|torch.Tensor,
                    )->torch.Tensor:
        # Calculates the gradient for all agents
        # INPUTS:
        #   parameter_instance: parameter(s) unique to your influence distribution
        # OUTPUTS: 
        #   grad: gradient matrix for all agents

        grad=0
        pr_matrix=self.prob_matrix(parameter_instance)
        if self.infl_type=='custom':
            d_matrix=self.d_torch(parameter_instance)
        elif self.infl_type in ['multi_gaussian','gaussian','Jones_M','dirl']:
            d_matrix=self.d_lnf_matrix(parameter_instance)
        pr_matrix_c=1-pr_matrix
        pr_prod=pr_matrix*pr_matrix_c
        if self.domain_type=='1d':
            if self.infl_fshift==True:
                shift_matrix=self.shift_matrix(parameter_instance)
            for a_id in range(self.num_agents):
                agent_grad=d_matrix[a_id]*pr_matrix[a_id]*pr_matrix_c[a_id]*torch.tensor(self.resource_distribution)
                if self.infl_fshift==True:
                    agent_grad=agent_grad-shift_matrix[a_id]*pr_matrix[a_id]*torch.tensor(self.resource_distribution)
                agent_grad=torch.sum(agent_grad)
                grad=matrix_builder(row_id=a_id,row=agent_grad,matrix=grad)
        else:
            for a_id in range(self.num_agents):
                agent_grad=d_matrix[a_id]*pr_prod[a_id]*torch.tensor(self.resource_distribution)
                agent_grad=torch.sum(agent_grad,1)
                grad=matrix_builder(row_id=a_id,row=agent_grad,matrix=grad)
            
    
        return grad
    
    def mv_gradient_accent(self,
                        show_out:bool=False,
                        grad_modify:bool=False,
                        reward:bool=True,
                        )->torch.Tensor:
        # Gradient Accent algorithm for multi variable games for players positons. Does all players at once.
        # INPUTS:
        #   show_out: returns the gradient and postion (reward),through vectors interations
        #   grad_motify: Special for figures only, it changes our grad to -1,0 for visualization or in simular matters
        #   reward: calculates and returns reward if output is True.  
        # OUTPUTS: 
        #   grad, postion, and possibly reward vectors for all agents through iterations of grad accent
        self.grad_modify=grad_modify
        pos_matrix=0
        grad_matrix=0
        reward_matrix=0
        reward_vec=0
        pos_vec=0
        grad_vec=0
        agents_og=self.agents_pos
        
        self.agents_pos=agents_og.copy()
        reward_vec=0
        pos_vec=0
        grad_vec=0
        for time in range(self.time_steps):
            grad_vec_row=self.gradient(self.parameters)
            if self.domain_type=='simplex':
                grad=torch.nn.functional.normalize(grad_vec_row,dim=1)
            else:
                grad=grad_vec_row
            temp=torch.tensor(np.array(self.agents_pos))+lr(iter=time,lr_type=self.lr_type,learning_rate=self.learning_rate)*grad
            for t_row in range(temp.size()[0]):
                if self.domain_type=='simplex':
                    temp_row=projection_onto_siplex(temp[t_row])
                    if torch.all(temp_row>0):
                        self.agents_pos[t_row]=temp_row.detach().numpy()
                    else:
                        pass
                else:
                    temp_row=temp[t_row]
                    self.agents_pos[t_row]=temp_row.detach().numpy()
            pos_vec_row=torch.tensor(np.array(self.agents_pos))
            pos_vec=matrix_builder(row_id=time,row=pos_vec_row,matrix=pos_vec)
            grad_vec=matrix_builder(row_id=time,row=grad_vec_row,matrix=grad_vec)
            
            if reward==True:
                reward_vec_row=self.reward_F(self.parameters)
                reward_vec=matrix_builder(row_id=time,row=reward_vec_row,matrix=reward_vec)
            if time>5:
                if self.domain_type=='simplex':
                    abs_diffrence=torch.linalg.norm(pos_vec_row-pos_vec[-5],axis=1)
                else:
                    abs_diffrence=torch.linalg.norm(pos_vec_row-pos_vec[-2],axis=1)
                abs_diffrence_value=torch.sum(abs_diffrence<=self.tolarance).item()
                if abs_diffrence_value>=self.tolarated_agents:
                    break
        self.grad_matrix=grad_vec.clone()
        self.pos_matrix=pos_vec.clone()
        if reward==True:
            self.reward_matrix=reward_vec
        
        
        
    
    def sv_gradient_accent(self,
                            show_out:bool=False,
                            grad_modify:bool=False,
                            reward:bool=True,
                            )->torch.Tensor:
        # Gradient Accent algorithm for single-variable games for players positons. Does all players at once.
        # INPUTS:
        #   show_out: returns the gradient and postion (reward),through vectors interations
        #   grad_motify: Special for figures only, it changes our grad to -1,0 for visualization or in simular matters
        #   reward: calculates and returns reward if output is True.  
        # OUTPUTS: 
        #   grad, postion, and possibly reward vectors for all agents through iterations of grad accent
        self.grad_modify=grad_modify
        reward_vec=0
        pos_vec=0
        grad_vec=0
        agents_og=self.agents_pos
        self.agents_pos=agents_og
        for time in range(self.time_steps):
            grad_vec_row=self.gradient(self.parameters)
            temp=torch.tensor(self.agents_pos)+lr(iter=time,lr_type=self.lr_type,learning_rate=self.learning_rate)*grad_vec_row
            
            for t_row in range(temp.size()[0]):
                temp_row=temp[t_row]
                if torch.all(temp_row>=self.domain_bounds[0]) and torch.all(temp_row<=self.domain_bounds[1]):
                    self.agents_pos[t_row]=temp_row.detach().numpy()
                else:
                    print("passed!")
                    pass
        
            pos_vec_row=torch.tensor(self.agents_pos)       
            pos_vec=matrix_builder(row_id=time,row=pos_vec_row,matrix=pos_vec)
            grad_vec=matrix_builder(row_id=time,row=grad_vec_row,matrix=grad_vec)
            if reward==True:
                reward_vec_row=self.reward_F(self.parameters)
                reward_vec=matrix_builder(row_id=time,row=reward_vec_row,matrix=reward_vec)
            if time>5:
                abs_diffrence=torch.abs(pos_vec_row-pos_vec[-2])
                abs_diffrence_value=torch.sum(abs_diffrence<=self.tolarance).item()
                if abs_diffrence_value>=self.tolarated_agents:
                    break
        self.grad_matrix=grad_vec
        self.pos_matrix=pos_vec
                        

        
    def gradient_accent(self,
                        show_out:bool=False,
                        grad_modify:bool=False,
                        reward:bool=True,
                        )->None|torch.Tensor:
        # Gradient Accent algorithm for our players. Does all players at once.
        # INPUTS:
        #   show_out: returns the gradient and postion (reward),through vectors interations
        #   grad_motify: Special for figures only, it changes our grad to -1,0 for visualization or in simular matters
        #   reward: calculates and returns reward if output is True.  
        # OUTPUTS: 
        #   grad, postion, and possibly reward vectors for all agents through iterations of grad accent
        
        agent_og=self.agents_pos
        
        if self.domain_type=='1d':
            self.sv_gradient_accent(show_out=show_out,grad_modify=grad_modify,reward=reward)
        else:
            self.mv_gradient_accent(show_out=show_out,grad_modify=grad_modify,reward=reward)
            
            
        self.agents_pos=agent_og
        if show_out==True:
            if reward==True:
                return self.pos_matrix, self.grad_matrix,self.reward_matrix
            else:
                return self.pos_matrix, self.grad_matrix

    

    def gradient_function(self,agents_pos,parameter_instance,ids=[0,1],two_a=True):
        # Calculates the gradient for all agents for any parameters
        # INPUTS:
        #   parameter_instance: parameter(s) unique to your influence distribution
        # OUTPUTS: 
        #   grad: gradient matrix for all agents

        grad=0
        og_pos=self.agents_pos
        self.agents_pos=agents_pos
        og_alpha=self.alpha_matrix
        pr_matrix=self.prob_matrix(parameter_instance)
        d_matrix=self.d_lnf_matrix(parameter_instance)
        pr_matrix_c=1-pr_matrix
        pr_prod=pr_matrix*pr_matrix_c
        if self.domain_type=='1d':
            if self.infl_fshift==True:
                shift_matrix=self.shift_matrix(parameter_instance)
            if two_a==False:
                agents=ids
            else:
                agents=range(self.num_agents)

            for a_id in agents:
                agent_grad=d_matrix[a_id]*pr_matrix[a_id]*pr_matrix_c[a_id]*torch.tensor(self.resource_distribution)
                if self.infl_fshift==True:
                    agent_grad=agent_grad-shift_matrix[a_id]*pr_matrix[a_id]*torch.tensor(self.resource_distribution)
                agent_grad=torch.sum(agent_grad)
                grad=matrix_builder(row_id=a_id,row=agent_grad,matrix=grad)
                self.alpha_matrix=og_alpha
        else:
            for a_id in range(self.num_agents):
                agent_grad=d_matrix[a_id]*pr_prod[a_id]*torch.tensor(self.resource_distribution)
                agent_grad=torch.sum(agent_grad,1)
                grad=matrix_builder(row_id=a_id,row=agent_grad,matrix=grad)
                self.alpha_matrix=og_alpha
        self.agents_pos=og_pos
        return grad
    
    