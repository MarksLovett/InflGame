import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import torch
from numpy import random

import hickle as hkl
from InflGame.adaptive.visualization import Shell
import InflGame.utils.general as general
import InflGame.domains.rd as rd
#The resource points
bin_points=np.linspace(.001, 0.999, 100)
alpha=.5
#Resource parameters
resource_parameters_gaussian=[[.1,.1],[.5-alpha/2,.5+alpha/2],[1,1]] #[[sd1, sd2,], [mean1,mean2], [factor1,factor2]]
#Resource distribution
resource_distribution2=rd.resource_distribution_choice(bin_points=bin_points,resource_type='multi_modal_gaussian_distribution_1D',resource_parameters=resource_parameters_gaussian)


domain_type='1d'
resource_distribution=resource_distribution2
mean=np.dot(bin_points,resource_distribution)/np.sum(resource_distribution) #mean of the resource distribution




num_agents=6 #number of agents
#int_agents_pos=general.agent_position_setup(num_agents=num_agents,setup_type='paper_default',domain_type=domain_type,domain_bounds=0)
#torch.tensor([.2,.21,0.33,0.66,.77,0.9])
#x_1=torch.tensor([0.9, 0.8, 0.7, 0.32, 0.1931, 0.0655]) #3,3 10k steps, 50% sig 0.03 to .3
x_1=torch.tensor([0.9255, 0.4816, 0.1969, 0.3216, 0.1931, 0.0655]) # 4,2 10k steps, 50% sig 0.03 to .3
#x_1=torch.tensor([0.0278, 0.1513, 0.0163, 0.8290, 0.3250, 0.3650]) # 5,1 alpha=.8 40k steps, 10% sig 0.03 to .4
int_agents_pos=x_1 #initial agent positions

infl_configs={"infl_type":"gaussian"} # influence type of the agents


parameters=general.agent_parameter_setup(num_agents=num_agents,infl_type=infl_configs["infl_type"],setup_type="initial_symmetric_setup",reach=.2) # parameters impacting agents reach (their std)
#parameters_custom=np.array([[.1,.2,.3,...]]) #needs to be length num_players


time_steps=1# number steps for the adaptive dynamics


vis=Shell(num_agents=num_agents,agents_pos=int_agents_pos,parameters=parameters,resource_distribution=resource_distribution,bin_points=bin_points, 
infl_configs = {'infl_type': 'gaussian'}, learning_rate_type= 'cosine_annealing', learning_rate= [.001, .001, 3000], time_steps=time_steps,
fp= 0, infl_cshift= False, cshift = 0, infl_fshift= False, Q = None,
domain_type = '1d', domain_bounds = [0, 1], resource_type = 'na', domain_refinement = 10,
tolerance = 10**-12, tolerated_agents = None,ignore_zero_infl= True)


vis.setup_adaptive_env()
vis.field.gradient_ascent()
og_pos_matrix=vis.field.pos_matrix
og_grad_matrix=vis.field.grad_matrix
vis.agents_pos=int_agents_pos.clone()
vis.field.agents_pos=int_agents_pos.clone()


def _bif_matrix_data(name='6_bif_sig_x_pos_x.hkl',matrix=None,load=False,save=False):
    if load:
        matrix=hkl.load(name)
        if type(matrix)==dict:
            matrix['max']=torch.tensor(matrix['max'])
            matrix['min']=torch.tensor(matrix['min'])
        else:
            matrix=torch.tensor(matrix)
    elif save:
        if matrix==None:
            raise TypeError('matrix to save cannot be None')
        if type(matrix)== dict:
            if torch.is_tensor(matrix['max']):
                matrix['max']=matrix['max'].numpy()
                matrix['min']=matrix['min'].numpy()
        else:
            matrix=matrix.numpy()
        hkl.dump(matrix,name)
    else:
        raise ValueError('Either load or save must be True')
    return matrix


        







st.write("Streamlit supports a wide range of data visualizations, including [Plotly, Altair, and Bokeh charts](https://docs.streamlit.io/develop/api-reference/charts). 📊 And with over 20 input widgets, you can easily make your data interactive!")

all_users = ["1", "2", "3"]
with st.container(border=True):
    users = st.multiselect("Users", all_users, default=all_users[0],max_selections=1)
name=r'demo\\paper_kernels\\Gaussian\\6p\\6_bif_pos_'+users[0]+'.hkl'
matrix=_bif_matrix_data(name=name,load=True)
fig,_=vis.equilibrium_bifurcation_plot(matrix=matrix,
                                        reach_start=.03,
                                        reach_end=.3,
                                        reach_num_points=200,
                                        time_steps=10000,
                                        plot_type="heat",
                                        name_ads=[],
                                        title_ads=[],
                                        refinements=10,
                                        parallel_configs={'parallel':True, 'max_workers':8, 'batch_size':4},
                                        font={'default_size': 20,'rect_label_size': 20, 'rect_sigma_size': 20, 'cbar_size': 22, 'title_size': 27, 'legend_size': 15, 'font_family': 'sans-serif'},
                                        cbar_config={'center_labels': True, 'label_alignment': 'center', 'shrink': 0.75},
                                        save=False,
                                        paper_figure={'paper': True, 'section': '3_2','figure_id':'fig1'},
                                        show_pred=False,
                                        optional_vline=None,
                                        envelope=True,
                                        complete=False,
                                        return_matrix=True,
                                        percentage=.5
                                        )
vis.field.pos_matrix=og_pos_matrix
vis.field.grad_matrix=og_grad_matrix


st.pyplot(fig)
