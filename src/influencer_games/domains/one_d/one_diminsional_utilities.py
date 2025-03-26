import numpy as np
import torch
import matplotlib.pyplot as plt
from random import randint
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


from influencer_games.utils.utilities import *



def color_list_maker(num_agents:int):
    colors_lst = []
    np.random.seed(1)
    for agent_color in range(num_agents):
        colors_lst.append('#%06X' % randint(0, 0xFFFFFF))
    return colors_lst

def critical_values_plot(num_agents:int,
                         bin_points:np.ndarray,
                         resource_distribution:torch.Tensor,
                         axis:plt.Axes,
                         reach_start:float=.3,
                         reach_end:float=0,
                         refinements=2
                         ):
    colors_lst=color_list_maker(num_agents=num_agents)
    num_sub_divisons=np.ceil(np.log2(num_agents))
    mean_divisions=[]
    mean_for_axis=[]
    std_divisions=[]
    for sub_division in range(int(num_sub_divisons)):
        if sub_division==0:
            mean_star=discrete_mean(bin_points=bin_points,resource_distribution=resource_distribution)
            std_star=np.sqrt((num_agents-2)/(num_agents-1)*discrete_variance(bin_points=bin_points,resource_distribution=resource_distribution,mean=mean_star))
            mean_divisions.append([mean_star])
            axis.axhline(mean_star,ls='--')
            axis.axvline(std_star,label=r'$t_*='+str(np.around(std_star,decimals=3))+r'$')
            std_divisions.append([std_star])
            mean_for_axis.append(mean_star)
        else:
            split_stds=[]
            symmetric_splits=[]
            split_new_means=[]
            mean_division=mean_divisions[sub_division-1]
            group_player_counts=split_favor_bottom(num_agents=num_agents,division=sub_division)
            for _ in range(refinements):
                symmetric_splits=symmetric_splitting(bin_points=bin_points,resource_distribution=resource_distribution,bifurication_count=sub_division,means=mean_division)[0]
                symmetric_splits.sort() 
                symmetric_splits=np.array(symmetric_splits)
                mid_point=(symmetric_splits[1:] + symmetric_splits[:-1]) / 2
                mean_division=mid_point
            for split_id in range(2**sub_division):
                if split_id==0:
                    support=np.where(mid_point[split_id]>bin_points)
                    split_new_means.append(mid_point[split_id])
                elif split_id==2**sub_division-1:
                    support=np.where(bin_points>mid_point[split_id-1])
                else:
                    split_new_means.append(mid_point[split_id])
                    a=mid_point[split_id]>bin_points
                    b=bin_points>mid_point[split_id-1]
                    support=np.where(a&b==True)
                values_supported=np.zeros(len(resource_distribution))
                values_supported[support]=resource_distribution[support].copy()
                mean_local=discrete_mean(bin_points=bin_points,resource_distribution=values_supported).copy()
                split_new_means.append(mean_local)
                if sub_division!=num_sub_divisons-1:
                    group_player_count=group_player_counts[split_id]
                    if group_player_count not in  [1,2]:
                        std_local=np.sqrt((group_player_count-2)/(group_player_count-1)*discrete_variance(bin_points=bin_points,resource_distribution=values_supported,mean=mean_local))
                        split_stds.append(std_local)
                    axis.axvline(std_local,label=r'$t_'+str(sub_division)+r'='+str(np.around(std_local,decimals=4))+r'$',color=colors_lst[sub_division])
                    axis.hlines(mean_local,xmin=reach_end,xmax=std_local,ls='--',color=colors_lst[sub_division])
                mean_for_axis.append(mean_local)
            split_stds=list(np.unique(np.around(split_stds,decimals=4)))
            if len(split_stds)>1:
                test=np.around(np.sum(split_stds-np.average(split_stds)),decimals=1)
                if test==0:
                    split_stds=[np.average(split_stds)]
            std_divisions.append(split_stds)
            mean_divisions.append(split_new_means)
            mean_divisions[sub_division].sort()
    return axis,mean_for_axis,std_divisions

def symmetric_splitting(bin_points,
                        resource_distribution,
                        bifurication_count,
                        means,
                        ):
    symmetric_splits=[]
    final_array=[]
    for split_id in range(2**bifurication_count):
        if split_id==0:
            support=np.where(means[split_id]>bin_points)
            final_array.append(means[split_id])
        elif split_id==2**bifurication_count-1:
            support=np.where(bin_points>means[split_id-1])
        else:
            final_array.append(means[split_id])
            a=means[split_id]>bin_points
            b=bin_points>means[split_id-1]
            support=np.where(a&b==True)
        values_supported=np.zeros(len(resource_distribution))
        values_supported[support]=resource_distribution[support].copy()
        mean_local=discrete_mean(bin_points=bin_points,resource_distribution=values_supported).copy()
        symmetric_splits.append(mean_local)
        final_array.append(mean_local)
    return symmetric_splits,final_array

def direction_strength_1d(gradient_function,
          two_a:bool,
          parameter_instance:list|np.ndarray|torch.Tensor = 0,
          ids:list = [0,1],
          pos:torch.Tensor = None,):
    Y, X = np.mgrid[0:1:100j, 0:1:100j]
    a1=X.flatten()
    a2=Y.flatten()
    if two_a==False:
        grads=[]
        for x,y in zip(a1,a2):
            pos[ids[0]]=x   
            pos[ids[1]]=y 
            grads.append(gradient_function(pos,parameter_instance,ids=ids,two_a=two_a).numpy())
        grads=np.array(grads)
    else:
        grads=np.array([gradient_function(np.array([x,y]),parameter_instance,ids=ids).numpy() for x,y in zip(a1,a2)])
    return grads


