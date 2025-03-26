import numpy
import sys
import os
import hickle as hkl
from pathlib import Path


def q_table_data_load(options)->dict:
    '''
    Reads the q-table from the specified path

    Args:
        optoins: dictionary with the following keys

    '''

    players=str(options['players'])+'_player'
    reach=options['reach']
    if reach=='small':
        sigma='small_sigma'
    elif reach=='large':
        sigma='large_sigma'
    else:
        sigma=None
    modes=str(options['modes'])+'m'
    densse=options['density']
    if densse==True:
        density='dense'
    else:
        density=None
    options=[players,sigma,modes,density]
    name_compoents=[]
    for option in options:
        if option!=None:
            name_compoents.append(option)
    
    folder=f"_".join(name_compoents)
    Path_Q=f"/".join(["data",players+"s",folder,"q_tables.hk1"])
    Path_C=f"/".join(["data",players+"s",folder,"configs.hk1"])
    q_table=hkl.load(Path_Q)
    configs=hkl.load(Path_C)
    
    return q_table, configs



def data_parmaters(configs:dict,data_type,resoure_type:str):
    if data_type=='q_tables' or data_type=='configs':
       data_parmater={'num_players':str(configs['env_config_main']['num_agents'])+'_players',
                      'data_type':data_type,
                      'reach':'sig_'+str(int(configs['env_config_main']['parameters'][0]*100)),
                      'resoure_type':resoure_type,
                      'steps':str(int(1/configs['env_config_main']['step_size']))+'_states',
        }
    return data_parmater
     


def data_directory(data_parameters,alt_name):
    my_path = os.path.dirname(os.path.abspath(__file__))
    cwd=my_path+'\\'+'data'
    p = Path(cwd)
    p.mkdir(exist_ok=True)

    file=[cwd,data_parameters['num_players']]
    file_name='\\'.join([str(x) for x in file ])
    p = Path(file_name)
    p.mkdir(exist_ok=True)
    
    for key in data_parameters.keys():
        if key!='data_type' and key!='num_players':
            file=file+[str(data_parameters[key])]
            file_name='\\'.join([str(x) for x in file ])
            p = Path(file_name)
            p.mkdir(exist_ok=True)
    if alt_name== True:
        file=file+[data_parameters['data_type']]
        file_name='\\'.join([str(x) for x in file ])
        p = Path(file_name)
        p.mkdir(exist_ok=True)
    
    return file_name

def data_name(data_parameters,name_ads,save_types):
    data_type=data_parameters['data_type']
    data_names=[]
    if data_type=='q_tables':
        data_name='q_table'
    elif data_type=='configs':
        data_name='configs'
    elif data_type=='reward_matrix':
        data_name='reward_matrix'

    if len(name_ads)>0:
        for name_additon in name_ads:
            data_name=data_name+'_'+name_additon
    for save_type in save_types:
        data_names.append(data_name+save_type)
    return data_names

def data_final_name(data_parameters,name_ads,save_types=['.hkl']):
    if data_parameters['data_type'] in ['nothingrn']:
        alt=True
    else:
        alt=False
    
    fig_names=data_name(data_parameters=data_parameters,name_ads=name_ads,save_types=save_types)
    file_names=[]
    for fig_name in fig_names:
        fig_direct=data_directory(data_parameters=data_parameters,alt_name=alt)
        file=[fig_direct,fig_name]
        file_name='\\'.join([str(x) for x in file ])
        file_names.append(file_name)
    return file_names