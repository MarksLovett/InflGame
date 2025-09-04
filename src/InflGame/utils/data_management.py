"""
.. module:: data_management
   :synopsis: Provides utilities for managing and organizing data in influencer games.

Data Management Utilities Module
================================

This module provides utility functions for managing and organizing data in influencer games. It includes functions 
for loading Q-tables, extracting data parameters, creating directory structures, and generating file names for saving 
and retrieving data.

Usage:
------
The `q_table_data_load` function loads Q-tables and configuration data, while the `data_directory` function creates 
directory structures based on data parameters. The `data_final_name` function generates full file paths for saving 
data.

Example:
--------

.. code-block:: python

    from InflGame.utils.data_management import q_table_data_load, data_final_name

    # Define options for loading Q-tables
    options = {
        "agents": 3,
        "reach": "small",
        "modes": 2,
        "density": True
    }

    # Load Q-table and configurations
    q_table, configs = q_table_data_load(options=options)
    print("Q-table loaded:", q_table)
    print("Configurations loaded:", configs)

    # Define data parameters
    data_parameters = {
        "num_agents": "3_agents",
        "data_type": "q_tables",
        "reach": "sig_50",
        "resource_type": "gaussian",
        "steps": "100_states"
    }

    # Generate final file names
    file_names = data_final_name(data_parameters=data_parameters, name_ads=["experiment1"], save_types=[".hkl"])
    print("Generated file names:", file_names)

"""

import numpy
import sys
import os
import hickle as hkl
from pathlib import Path
from typing import Dict, List, Union, Optional

 
def q_table_data_load(options: Dict[str, Union[str, int, bool]]
                      ) -> Dict[str, Union[dict, list]]:
    """
    Reads the Q-table and configuration data from specified paths.

    :param options: A dictionary containing the following keys:
        - ``agents`` (str): Number of agents.
        - ``reach`` (str): Reach type ('small' or 'large').
        - ``modes`` (str): Modes of operation.
        - ``density`` (bool): Whether the data is dense.
    :type options: dict
    :return: A tuple containing the Q-table and configuration data.
    :rtype: dict
    """
    agents=str(options['agents'])+'_agent'
    reach=options['reach']
    if reach=='small':
        sigma='small_sigma'
    elif reach=='large':
        sigma='large_sigma'
    else:
        sigma=None
    modes=str(options['modes'])+'m'
    dense=options['density']
    if dense==True:
        density='dense'
    else:
        density=None
    options=[agents,sigma,modes,density]
    name_components=[]
    for option in options:
        if option!=None:
            name_components.append(option)
    
    folder=f"_".join(name_components)
    Path_Q=f"/".join(["data",agents+"s",folder,"q_tables.hk1"])
    Path_C=f"/".join(["data",agents+"s",folder,"configs.hk1"])
    q_table=hkl.load(Path_Q)
    configs=hkl.load(Path_C)
    
    return q_table, configs



def data_parameters(configs: Dict[str, dict],
                   data_type: str,
                   resource_type: str
                   ) -> Optional[Dict[str, str]]:
    """
    Extracts data parameters from the configuration file.

    :param configs: Configuration dictionary.
    :type configs: dict
    :param data_type: Type of data ('q_tables' or 'configs').
    :type data_type: str
    :param resource_type: Type of resource.
    :type resource_type: str
    :return: A dictionary containing data parameters.
    :rtype: dict
    """
    if data_type in ['q_tables','configs',"final_mad","final_positions"]:
       data_parameter={'num_agents':str(configs['env_config_main']['num_agents'])+'_agents',
                      'data_type':data_type,
                      'reach':'sig_'+str(int(configs['env_config_main']['parameters'][0]*100)),
                      'resource_type':resource_type,
                      'steps':str(int(1/configs['env_config_main']['step_size']))+'_states',
        }
    return data_parameter
     


def data_directory(data_parameters: Dict[str, str],
                   alt_name: bool,
                   paper_figure: bool = False
                   ) -> str:
    """
    Creates a directory structure based on data parameters.

    :param data_parameters: Dictionary containing data parameters.
    :type data_parameters: dict
    :param alt_name: Whether to use an alternative name.
    :type alt_name: bool
    :return: The final directory path.
    :rtype: str
    """
    if data_parameters['data_type'] in ["plot"]:
        if paper_figure==True:
            my_path = os.path.dirname(os.path.abspath(__file__))
            cwd=my_path+'\\'+'paper_plots'
            p = Path(cwd)
            p.mkdir(exist_ok=True)
        else:
            my_path = os.path.dirname(os.path.abspath(__file__))
            cwd=my_path+'\\'+'plots'
            p = Path(cwd)
            p.mkdir(exist_ok=True)
    else:
        my_path = os.path.dirname(os.path.abspath(__file__))
        cwd=my_path+'\\'+'data'
        p = Path(cwd)
        p.mkdir(exist_ok=True)
    if paper_figure==True:
        file=[cwd,data_parameters['section']]
        file_name='\\'.join([str(x) for x in file ])
        p = Path(file_name)
        p.mkdir(exist_ok=True)


        file=file+[data_parameters['figure_id']]
        file_name='\\'.join([str(x) for x in file ])
        p = Path(file_name)
        p.mkdir(exist_ok=True)
        return file_name
    else:
        if data_parameters['data_type'] in ["plots"]:
            file=[cwd,data_parameters['domain_type']]
            file_name='\\'.join([str(x) for x in file ])
            p = Path(file_name)
            p.mkdir(exist_ok=True)
            for key in data_parameters.keys():
                if key!='data_type' and key!='domain_type':
                    file=file+[str(data_parameters[key])]
                    file_name='\\'.join([str(x) for x in file ])
                    p = Path(file_name)
                    p.mkdir(exist_ok=True)

        elif data_parameters['data_type'] in ["final_positions", "final_mad"]:
            file=[cwd,data_parameters['num_agents']]
            file_name='\\'.join([str(x) for x in file ])
            p = Path(file_name)
            p.mkdir(exist_ok=True)
            for key in ['bifurcation']:
                file=file+[key]
                file_name='\\'.join([str(x) for x in file ])
                p = Path(file_name)
                p.mkdir(exist_ok=True)

        else:
            file=[cwd,data_parameters['num_agents']]
            file_name='\\'.join([str(x) for x in file ])
            p = Path(file_name)
            p.mkdir(exist_ok=True)
            for key in data_parameters.keys():
                if key!='data_type' and key!='num_agents':
                    file=file+[str(data_parameters[key])]
                    file_name='\\'.join([str(x) for x in file ])
                    p = Path(file_name)
                    p.mkdir(exist_ok=True)
        return file_name

def data_name(data_parameters: Dict[str, str],
              name_ads: List[str],
              save_types: List[str],
              paper_figure: bool = False
              ) -> List[str]:
    """
    Generates data file names based on parameters and additional names.

    :param data_parameters: Dictionary containing data parameters.
    :type data_parameters: dict
    :param name_ads: List of additional name components.
    :type name_ads: list
    :param save_types: List of file extensions.
    :type save_types: list
    :return: A list of generated file names.
    :rtype: list
    """
    data_type=data_parameters['data_type']
    data_names=[]
    if data_type=='q_tables':
        data_name='q_table'
    elif data_type=='configs':
        data_name='configs'
    elif data_type=='reward_matrix':
        data_name='reward_matrix'
    elif data_type=='mean_positions':
        data_name='mean_positions'
    elif data_type=='MAD':
        data_name='MAD'
    elif data_type=='final_positions':
        data_name='final_positions'
    elif data_type=='final_mad':
        data_name='final_mad'
    elif data_type=='plot':
        if paper_figure==True:
            agents=str(data_parameters['num_agents'])
            data_name=data_parameters['domain_type']+'_'+data_parameters['plot_type']+'_'+agents+'_agents'
        else:
            data_name=data_parameters["plot_type"]

        
    else:
        raise ValueError(f"Unknown data type: {data_type}")



    if len(name_ads)>0:
        for name_addition in name_ads:
            data_name=data_name+'_'+name_addition
    for save_type in save_types:
        data_names.append(data_name+save_type)
    return data_names

def data_final_name(data_parameters: Dict[str, str],
                    name_ads: List[str],
                    save_types: List[str] = ['.hkl'],
                    paper_figure: bool = False
                    ) -> List[str]:
    """
    Generates the final file names with directory paths.

    :param data_parameters: Dictionary containing data parameters.
    :type data_parameters: dict
    :param name_ads: List of additional name components.
    :type name_ads: list
    :param save_types: List of file extensions (default is ['.hkl']).
    :type save_types: list
    :return: A list of full file paths.
    :rtype: list
    """
    if data_parameters['data_type'] in ['nothingrn']:
        alt=True
    else:
        alt=False
    
    data_names=data_name(data_parameters=data_parameters,name_ads=name_ads,save_types=save_types,paper_figure=paper_figure)
    file_names=[]
    for data_sy in data_names:
        data_direct=data_directory(data_parameters=data_parameters,alt_name=alt,paper_figure=paper_figure)
        file=[data_direct,data_sy]
        file_name='\\'.join([str(x) for x in file ])
        file_names.append(file_name)
    return file_names
