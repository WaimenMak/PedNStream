# -*- coding: utf-8 -*-
# @Time    : 26/01/2025 15:30
# @Author  : mmai
# @FileName: __init__
# @Software: PyCharm

"""
Reinforcement Learning module for PedNStream crowd control.

This module provides PettingZoo multi-agent environment wrappers for training
RL agents to control pedestrian traffic flow through:
- Separators: bidirectional lane allocation via separator_width
- Gaters: node-level gate control via front_gate_width
"""

from .pz_pednet_env import PedNetParallelEnv

__all__ = ['PedNetParallelEnv']

