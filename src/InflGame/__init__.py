
"""
InflGame: A influencer game modeling package
============================================================


Subpackages
-----------
Using any of these subpackages requires an explicit import. For example,
``import influencer_games.adaptive_dynamics``.

::

 adaptive                           --- Adaptive dynamics for influencer games and their visualization
 domains                            --- Domains for the use in both adaptive and RL dynamics
 kernels                            --- Influence_kernels from the papers
 MARL                               --- Reinforcement learning modules
 utils                              --- Utilities modules for across the package
 
 


"""


submodules = [
    'adaptive',
    'domains',
    'kernels',
    'MARL',
    'utils',
]

__all__ = submodules



