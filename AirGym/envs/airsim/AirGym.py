# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:16:10 2017

@author: Kjell
"""
import logging
import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box

from envs.airsim.myAirSimMultirotorClient import *
        
from AirSimClient import *

logger = logging.getLogger(__name__)

class AirSimMultirotorEnv(gym.Env):

    airgym = None
        
    def __init__(self):
        # left depth, center depth, right depth, distance, x_val, y_val
        self.low = np.array([   0.0,   0.0,   0.0,   0.0,   -2.0, -74.0 ])
        self.high = np.array([100.0, 100.0, 100.0, 239.37, 227.0,  78.0])
        self.observation_space = spaces.Box(self.low, self.high)
        self.action_space = spaces.Discrete(4)
		
        self.goal = 	[221.0, -9.0] # global xy coordinates
        
        self.state = (100.0, 100.0, 100.0, 221.0, 0.0, 0.0)
        
        self.episodeN = 0
        self.stepN = 0 
        
        self.allLogs = { 'reward':[0] }
        self.allLogs['distance'] = [221]
        self.allLogs['action'] = [1]

        
        self._seed()
        self.stallCount = 0
        global airgym
        airgym = myAirSimMultirotorClient()
        

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def computeReward(self, now):
	
		# test if getPosition works here liek that
		# get exact coordiantes of the tip
      
        distance_now = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        distance_before = self.allLogs['distance'][-1]
        
        if abs(distance_now - distance_before) < 0.0001:
            
            if self.stepN > 3 and abs(self.allLogs['distance'][-1]  -  self.allLogs['distance'][-2]) == 0:
                return -100.0, distance_now
            
            else: return -1.0, distance_now
            
        else: return distance_before - distance_now, distance_now
		

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.addToLog('action', action)
        
        self.stepN += 1

        collided = airgym.take_action(action)
        
        now = airgym.getPosition()
                 	   
        if collided == True:
            done = True
            reward = -100.0
            distance = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        else: 
            done = False
            reward, distance = self.computeReward(now)
    
        self.sensors = airgym.getSensorStates()
        self.state = self.sensors
    
        #append distance, x_val, y_val to state
        self.state.append(distance)
        self.state.append(now.x_val)
        self.state.append(now.y_val)
        
        # Training using the Roaming mode 
        self.addToLog('reward', reward)
        rewardSum = np.sum(self.allLogs['reward'])
        
        self.addToLog('distance', distance)
        

        # Terminate the episode on large cumulative amount penalties, 
        # since drone probably got into an unexpected loop of some sort
        if rewardSum < -1000:
            done = True
        
        # Youuuuu made it
        if distance < 3:
            done = True
            reward = 100.0
            
        sys.stdout.write("\r\x1b[K{}/{}==>reward/depth: {:.1f}/{:.1f}   \t({:.1f}/{:.1f}/{:.1f})   {:.0f}".format(self.episodeN, self.stepN, reward, rewardSum, self.state[0], self.state[1], self.state[2], action))
        sys.stdout.flush()

        return np.array(self.state), reward, done, {}

    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)
        
    def _reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        
        airgym.reset()
        self.stepN = 0
        self.episodeN += 1
        
        self.allLogs = { 'reward': [0] }
        self.allLogs['distance'] = [221]
        self.allLogs['action'] = [1]
        
        print("")
        
        self.sensors = airgym.getSensorStates()
        
        # Initial state
        self.state = (self.sensors[0], self.sensors[0], self.sensors[0], 221.0, 0.0, 0.0)
        return np.array(self.state)
