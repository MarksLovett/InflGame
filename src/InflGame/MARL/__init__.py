
"""
MARL (:mod:`InflGame.MARL`)
==============================================================

.. currentmodule:: InflGame.MARL

Models specific to multi-agent reinforcement learning. 

Synchronous IQL (independent Q-Learning)
===========================================

Players take actions at the same time 

==================  =============================================
Submodule           Description
==================  =============================================
`sync_game`         The environment for the learning agents.
`IQL_sync`          How the agents learn via asynchronous IQL
==================  =============================================

Asynchronous IQL (independent Q-Learning)
===========================================

Players take actions at the same time 

==================  =============================================
Submodule           Description
==================  =============================================
`async_game`         The environment for the learning agents.
`IQL_async`          How the agents learn via asynchronous IQL
==================  =============================================
    
Utility/Misc
===========================================

Modules for plotting and running experiments

==================  ================================================================
Submodule           Description
==================  ================================================================
`MARL_plots`         Plotting  functions for visualizing the results of IQL methods.
`utils`              Utilities for the MARL dynamics
==================  ================================================================

    
   
"""