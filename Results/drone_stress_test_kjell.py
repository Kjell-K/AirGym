# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:15:29 2017

@author: Kjell
"""

import time
import math

from AirSimClient import *

# connect to the AirSim simulator 
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


def straight(duration, speed):
    pitch, roll, yaw  = client.getPitchRollYaw()
    vx = math.cos(yaw) * speed
    vy = math.sin(yaw) * speed
    client.moveByVelocityZ(vx, vy, -6, duration, DrivetrainType.ForwardOnly)
    start = time.time()
    return start, duration

def take_action():
		
    start = time.time()
    duration = 0 
    
    collided = False
		  
    start, duration = straight(5, 4)  # for 5 sec with "speed" 4 or until it collides  

    while duration > time.time() - start:
        if client.getCollisionInfo().has_collided == True:
            client.moveByVelocity(0, 0, 0, 1)
            return True    
    return collided


def reset():
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.moveToZ(-6, 3) 
    time.sleep(3)
    
    
if __name__ == "__main__":
    reset()
    for idx in range(250000): #250k
        collided = take_action()
        if collided == True:
            reset()
        print("%d" % idx)
    
    # that's enough fun for now. let's quite cleanly
    client.enableApiControl(False)
