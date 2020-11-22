import logging
import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box

from gym_airsim.envs.myAirSimClient import *
        

logger = logging.getLogger(__name__)


class AirSimEnv(gym.Env):

    airgym = None
        
    def __init__(self):
        # left depth, center depth, right depth, yaw
        self.observation_space = spaces.Box(low=0, high=255, shape=(30, 100))
        self.state = np.zeros((30, 100), dtype=np.uint8)  
        
        self.action_space = spaces.Discrete(3)
		
        self.goal = 	[221.0, -9.0] # global xy coordinates
        
        
        self.episodeN = 0
        self.stepN = 0 
        
        self.allLogs = { 'reward':[0] }
        self.allLogs['distance'] = [221]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]


        self._seed()
        
        global airgym
        airgym = myAirSimClient()
        
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def computeReward(self, now, track_now):
	
		# test if getPosition works here liek that
		# get exact coordiantes of the tip
      
        distance_now = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        
        distance_before = self.allLogs['distance'][-1]
              
        r = -1
        
        """
        if abs(distance_now - distance_before) < 0.0001:
            r = r - 2.0
            #Check if last 4 positions are the same. Is the copter actually moving?
            if self.stepN > 5 and len(set(self.allLogs['distance'][len(self.allLogs['distance']):len(self.allLogs['distance'])-5:-1])) == 1: 
                r = r - 50
        """  
            
        r = r + (distance_before - distance_now)
            
        return r, distance_now
		
    
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.addToLog('action', action)
        
        self.stepN += 1

        collided = airgym.take_action(action)
        
        now = airgym.getPosition()
        track = airgym.goal_direction(self.goal, now) 

        if collided == True:
            done = True
            reward = -100.0
            distance = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        elif collided == 99:
            done = True
            reward = 0.0
            distance = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        else: 
            done = False
            reward, distance = self.computeReward(now, track)
        
        # Youuuuu made it
        if distance < 3:
            done = True
            reward = 100.0
        
        self.addToLog('reward', reward)
        rewardSum = np.sum(self.allLogs['reward'])
        self.addToLog('distance', distance)
        self.addToLog('track', track)      
            
        # Terminate the episode on large cumulative amount penalties, 
        # since drone probably got into an unexpected loop of some sort
        if rewardSum < -100:
            done = True
        
        sys.stdout.write("\r\x1b[K{}/{}==>reward/depth: {:.1f}/{:.1f}   \t {:.0f}  {:.0f}".format(self.episodeN, self.stepN, reward, rewardSum, track, action))
        sys.stdout.flush()
        
        info = {"x_pos" : now.x_val, "y_pos" : now.y_val}
        self.state = airgym.getScreenDepthVis(track)

        return self.state, reward, done, info

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
        airgym.AirSim_reset()
        
        self.stepN = 0
        self.episodeN += 1
        
        self.allLogs = { 'reward': [0] }
        self.allLogs['distance'] = [221]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]
        
        print("")
        
        now = airgym.getPosition()
        track = airgym.goal_direction(self.goal, now)
        self.state = airgym.getScreenDepthVis(track)
        
        return self.state