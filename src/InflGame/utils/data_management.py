"""
.. module:: data_management
   :synopsis: Provides utilities for managing and organizing data files in influencer games experiments.

Data Management Utilities Module
================================

This module provides comprehensive utility functions for managing and organizing data in influencer games research.
It handles Q-table loading, data parameter extraction, hierarchical directory structure creation, and standardized
file naming conventions for saving and retrieving experimental results.

The module supports multiple data types including Q-tables, configuration files, reward matrices, position data,
and mean absolute deviation (MAD) metrics. It automatically creates organized directory hierarchies based on
experiment parameters such as number of agents, influence reach, resource types, and state discretization.

Dependencies:
-------------
- numpy: Array operations
- hickle: HDF5-based serialization for Python objects
- pathlib: Object-oriented filesystem paths
- typing: Type hints support

Key Functions:
--------------
- q_table_data_load: Load Q-tables and configurations from standardized paths
- data_parameters: Extract and format data parameters from configuration dictionaries
- data_directory: Create hierarchical directory structures for data organization
- data_name: Generate standardized file names based on experiment parameters
- data_final_name: Combine directory paths and file names for complete file paths

Usage:
------
The typical workflow involves defining experiment options, loading existing data, extracting parameters,
and generating standardized paths for saving new results. The module enforces consistent naming conventions
across all influencer games experiments.

Example:
--------

.. code-block:: python

    from InflGame.utils.data_management import q_table_data_load, data_final_name
    import hickle as hkl

    # Load existing Q-tables and configurations
    options = {
        "agents": 3,
        "reach": "small",
        "modes": 2,
        "density": True
    }
    q_table, configs = q_table_data_load(options=options)
    print(f"Q-table shape: {q_table.shape}")
    
    # Generate standardized file paths for saving new data
    data_params = {
        "num_agents": "3_agents",
        "data_type": "q_tables",
        "reach": "sig_50",
        "resource_type": "gaussian",
        "steps": "100_states"
    }
    file_paths = data_final_name(
        data_parameters=data_params,
        name_ads=["experiment1", "trial1"],
        save_types=[".hkl", ".npy"]
    )
    
    # Save data using generated paths
    hkl.dump(q_table, file_paths[0])

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
    Load Q-table and configuration data from standardized file paths.
    
    This function constructs file paths based on experiment options and loads
    pre-computed Q-tables and configuration dictionaries from HDF5-based hickle files.
    The path structure follows the convention: ``data/{agents}/{folder}/q_tables.hkl``
    where folder is constructed from agent count, reach parameter, modes, and density.
    
    Parameters
    ----------
    options : Dict[str, Union[str, int, bool]]
        Dictionary containing experiment configuration with required keys:
        
        - ``agents`` : int
            Number of agents in the multi-agent system
        - ``reach`` : str
            Influence reach parameter, either ``'small'`` or ``'large'``
        - ``modes`` : int
            Number of operational modes in the environment
        - ``density`` : bool
            Whether the resource distribution is dense
    
    Returns
    -------
    Tuple[dict, dict]
        A tuple containing:
        
        - **q_table** : dict
            Loaded Q-table data structure mapping states to action values
        - **configs** : dict
            Configuration dictionary containing environment parameters
    
    Raises
    ------
    FileNotFoundError
        If Q-table or configuration files do not exist at constructed paths
    ValueError
        If ``reach`` parameter is not ``'small'`` or ``'large'``
    
    Notes
    -----
    The function maps reach values to sigma parameters:
    
    - ``'small'`` → ``'small_sigma'``
    - ``'large'`` → ``'large_sigma'``
    
    File naming convention follows the pattern:
    ``{agents}_agents_{sigma}_{modes}m_{density}/q_tables.hkl``
    
    Examples
    --------
    Load Q-tables for a 3-agent system with small reach:
    
    >>> options = {
    ...     "agents": 3,
    ...     "reach": "small",
    ...     "modes": 2,
    ...     "density": True
    ... }
    >>> q_table, configs = q_table_data_load(options=options)
    >>> print(f"Loaded Q-table type: {type(q_table)}")
    Loaded Q-table type: <class 'dict'>
    
    Load data for large reach parameter:
    
    >>> options = {"agents": 5, "reach": "large", "modes": 3, "density": False}
    >>> q_table, configs = q_table_data_load(options)
    
    """
    agents = str(options['agents']) + '_agent'
    reach = options['reach']
    if reach == 'small':
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
    Extract and format data parameters from configuration dictionary.
    
    This function parses experiment configurations and extracts key parameters including
    agent count, influence reach, resource type, and state discretization. It formats
    these parameters into a standardized dictionary suitable for file naming and directory
    structure generation.
    
    Parameters
    ----------
    configs : Dict[str, dict]
        Configuration dictionary containing experiment parameters. Must have an
        ``'env_config_main'`` key with nested parameters including:
        
        - ``'num_agents'`` : int
            Number of agents in the system
        - ``'parameters'`` : list or array
            Influence parameters (first element used for reach)
        - ``'step_size'`` : float
            State discretization step size
    
    data_type : str
        Type of data being processed. Supported values:
        
        - ``'q_tables'`` : Q-learning tables
        - ``'configs'`` : Configuration files
        - ``'final_mad'`` : Final mean absolute deviation
        - ``'final_positions'`` : Final agent positions
    
    resource_type : str
        Type of resource distribution (e.g., ``'gaussian'``, ``'uniform'``, ``'beta'``)
    
    Returns
    -------
    Optional[Dict[str, str]]
        Dictionary containing formatted parameters with keys:
        
        - ``'num_agents'`` : str
            Formatted as ``'{N}_agents'``
        - ``'data_type'`` : str
            The input data type
        - ``'reach'`` : str
            Formatted as ``'sig_{value}'`` where value is :math:`100 \\times \\sigma`
        - ``'resource_type'`` : str
            The input resource type
        - ``'steps'`` : str
            Number of discrete states, formatted as ``'{N}_states'``
        
        Returns ``None`` if ``data_type`` is not in supported types.
    
    Notes
    -----
    The reach parameter is computed as:
    
    .. math::
        \\text{reach} = \\lfloor 100 \\times \\sigma \\rfloor
    
    where :math:`\\sigma` is the first element of ``configs['env_config_main']['parameters']``.
    
    The number of states is computed as:
    
    .. math::
        \\text{states} = \\lfloor 1 / \\text{step_size} \\rfloor
    
    Examples
    --------
    Extract parameters from a standard configuration:
    
    >>> configs = {
    ...     'env_config_main': {
    ...         'num_agents': 3,
    ...         'parameters': [0.5, 0.3],
    ...         'step_size': 0.01
    ...     }
    ... }
    >>> params = data_parameters(configs, 'q_tables', 'gaussian')
    >>> print(params)
    {'num_agents': '3_agents', 'data_type': 'q_tables', 'reach': 'sig_50', 
     'resource_type': 'gaussian', 'steps': '100_states'}
    
    """
    if data_type in ['q_tables', 'configs', "final_mad", "final_positions"]:
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
    Create hierarchical directory structure for organized data storage.
    
    This function builds a nested directory hierarchy based on experiment parameters,
    automatically creating all necessary parent directories. It supports different
    organizational schemes for research data, plots, and publication-ready figures.
    
    Parameters
    ----------
    data_parameters : Dict[str, str]
        Dictionary containing organizational parameters. Required keys vary by ``data_type``:
        
        - For plots: ``'data_type'``, ``'section'``, ``'figure_id'`` (if paper_figure=True)
        - For data: ``'data_type'``, ``'num_agents'``, ``'reach'``, ``'resource_type'``, ``'steps'``
    
    alt_name : bool
        Whether to use alternative naming scheme (currently unused, reserved for future use)
    
    paper_figure : bool, optional
        If ``True``, creates directory structure for publication figures organized by
        section and figure ID. Default is ``False``.
    
    Returns
    -------
    str
        Absolute path to the created directory with Windows path separators (``\\``)
    
    Notes
    -----
    Directory structure patterns:
    
    **For paper figures** (``paper_figure=True``):
    
    .. code-block:: text
    
        {module_path}/paper_plots/{section}/{figure_id}/
    
    **For regular plots** (``data_type='plot'``):
    
    .. code-block:: text
    
        {module_path}/plots/{domain_type}/{param1}/{param2}/...
    
    **For data files**:
    
    .. code-block:: text
    
        {module_path}/data/{num_agents}/{param1}/{param2}/...
    
    All intermediate directories are created automatically using ``pathlib.Path.mkdir(exist_ok=True)``.
    
    Examples
    --------
    Create directory for paper figure:
    
    >>> params = {
    ...     'data_type': 'plot',
    ...     'section': 'results',
    ...     'figure_id': 'fig_1'
    ... }
    >>> path = data_directory(params, alt_name=False, paper_figure=True)
    >>> print(path)
    C:\\...\\paper_plots\\results\\fig_1
    
    Create directory for Q-table data:
    
    >>> params = {
    ...     'data_type': 'q_tables',
    ...     'num_agents': '3_agents',
    ...     'reach': 'sig_50',
    ...     'resource_type': 'gaussian'
    ... }
    >>> path = data_directory(params, alt_name=False)
    >>> print(path)
    C:\\...\\data\\3_agents\\sig_50\\gaussian
    
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
    Generate standardized file names based on data type and parameters.
    
    This function creates descriptive file names following consistent naming conventions
    for different data types. It supports multiple file formats and allows appending
    custom suffixes for experiment versioning and identification.
    
    Parameters
    ----------
    data_parameters : Dict[str, str]
        Dictionary containing data parameters. Required keys:
        
        - ``'data_type'`` : str
            Type of data (``'q_tables'``, ``'configs'``, ``'plot'``, etc.)
        
        For plots, additional keys:
        
        - ``'plot_type'`` : str
            Type of plot visualization
        - ``'domain_type'`` : str
            Domain type (``'1d'``, ``'2d'``, ``'simplex'``)
        - ``'num_agents'`` : str
            Number of agents (if ``paper_figure=True``)
    
    name_ads : List[str]
        List of additional name components to append (e.g., experiment IDs, trial numbers).
        Components are joined with underscores.
    
    save_types : List[str]
        List of file extensions including the dot (e.g., ``['.hkl', '.npy', '.png']``)
    
    paper_figure : bool, optional
        If ``True``, uses publication naming format for plots. Default is ``False``.
    
    Returns
    -------
    List[str]
        List of complete file names, one for each save type. Each name combines
        the base name, additional components, and file extension.
    
    Raises
    ------
    ValueError
        If ``data_type`` is not recognized
    
    Notes
    -----
    Base name mapping by data type:
    
    - ``'q_tables'`` → ``'q_table'``
    - ``'configs'`` → ``'configs'``
    - ``'reward_matrix'`` → ``'reward_matrix'``
    - ``'mean_positions'`` → ``'mean_positions'``
    - ``'MAD'`` → ``'MAD'``
    - ``'final_positions'`` → ``'final_positions'``
    - ``'final_mad'`` → ``'final_mad'``
    - ``'plot'`` → custom format based on plot parameters
    
    For paper figures, plot names follow:
    ``{domain_type}_{plot_type}_{num_agents}_agents``
    
    Examples
    --------
    Generate Q-table file names with multiple formats:
    
    >>> params = {'data_type': 'q_tables'}
    >>> names = data_name(params, name_ads=['exp1', 'v2'], save_types=['.hkl', '.npy'])
    >>> print(names)
    ['q_table_exp1_v2.hkl', 'q_table_exp1_v2.npy']
    
    Generate paper figure name:
    
    >>> params = {
    ...     'data_type': 'plot',
    ...     'domain_type': '2d',
    ...     'plot_type': 'bifurcation',
    ...     'num_agents': '3'
    ... }
    >>> names = data_name(params, name_ads=[], save_types=['.png'], paper_figure=True)
    >>> print(names)
    ['2d_bifurcation_3_agents.png']
    
    """
    data_type = data_parameters['data_type']
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
    Generate complete file paths combining directory structure and file names.
    
    This function is the primary interface for generating standardized file paths in the
    influencer games framework. It combines directory creation (via ``data_directory``)
    and file naming (via ``data_name``) into complete absolute paths ready for saving
    or loading data.
    
    Parameters
    ----------
    data_parameters : Dict[str, str]
        Dictionary containing all necessary parameters for path construction.
        Required keys depend on ``data_type`` (see ``data_directory`` and ``data_name``
        for specific requirements).
    
    name_ads : List[str]
        List of additional descriptive components to append to file names.
        Useful for experiment versioning, trial IDs, or custom identifiers.
    
    save_types : List[str], optional
        List of file extensions including dots. Default is ``['.hkl']`` (hickle format).
        Common options: ``['.hkl', '.npy', '.pkl', '.png', '.svg']``
    
    paper_figure : bool, optional
        If ``True``, generates paths for publication-ready figures with special
        directory organization. Default is ``False``.
    
    Returns
    -------
    List[str]
        List of complete absolute file paths, one for each save type.
        Paths use Windows separators (``\\``) and include all directory components.
    
    Notes
    -----
    This function ensures all necessary directories exist before returning paths.
    The directory creation is handled internally by ``data_directory``.
    
    Path structure follows:
    
    .. code-block:: text
    
        {base_dir}/{param1}/{param2}/.../{base_name}_{ad1}_{ad2}{ext}
    
    Examples
    --------
    Generate paths for Q-table storage:
    
    >>> params = {
    ...     'num_agents': '3_agents',
    ...     'data_type': 'q_tables',
    ...     'reach': 'sig_50',
    ...     'resource_type': 'gaussian',
    ...     'steps': '100_states'
    ... }
    >>> paths = data_final_name(params, name_ads=['exp1'], save_types=['.hkl'])
    >>> print(paths[0])
    C:\\...\\data\\3_agents\\sig_50\\gaussian\\100_states\\q_table_exp1.hkl
    
    Generate multiple format paths for plots:
    
    >>> params = {
    ...     'data_type': 'plot',
    ...     'plot_type': 'bifurcation',
    ...     'domain_type': '1d',
    ...     'num_agents': '3'
    ... }
    >>> paths = data_final_name(params, name_ads=['trial1'], 
    ...                         save_types=['.png', '.svg'])
    >>> len(paths)
    2
    
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
