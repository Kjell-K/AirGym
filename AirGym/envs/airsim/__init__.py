# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:16:10 2017

@author: Kjell
"""

# ref: https://github.com/openai/gym/issues/626
from gym.envs.registration import register

"""
register(
    id='AirSimMultirotorEnv-v41',
    entry_point='envs.airsim.AirGym:AirSimMultirotorEnv',
    max_episode_steps=200,
    reward_threshold=300,
)
"""