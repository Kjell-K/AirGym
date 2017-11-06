# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:16:10 2017

@author: Kjell
"""
import numpy as np
import time
import math
from desktopmagic.screengrab_win32 import (getDisplayRects, getRectAsImage)
import cv2

# Change the path below to point to the directoy where you installed the AirSim PythonClient
#sys.path.append('C:/Users/Kjell/Google Drive/MASTER-THESIS/AirSimpy')

from AirSimClient import *

client = MultirotorClient()

class myAirSimMultirotorClient(MultirotorClient):

    def __init__(self):        
        self.img1 = None
        self.img2 = None

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.enableApiControl(True)
        self.armDisarm(True)
    
        self.home_pos = self.getPosition()
    
        self.home_ori = self.getOrientation()
        
        self.z = -6
    
    def straight(self, duration, speed):
        pitch, roll, yaw  = self.getRollPitchYaw()
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        self.moveByVelocityZ(vx, vy, self.z, duration, DrivetrainType.ForwardOnly)
        start = time.time()
        return start, duration

    def go_left(self, duration):
        pitch, roll, yaw  = self.getRollPitchYaw()
        vx = math.cos(yaw)
        vy = math.sin(yaw)
        self.moveByVelocityZ(vx, vy, self.z, duration, drivetrain = DrivetrainType.ForwardOnly, yaw_mode = YawMode(False,0))
        start = time.time()
        return start, duration   
 
    def go_right(self, duration):
        pitch, roll, yaw  = self.getRollPitchYaw()
        vx = math.cos(yaw)
        vy = math.sin(yaw)
        self.moveByVelocityZ(vx, vy, self.z, duration, drivetrain = DrivetrainType.ForwardOnly, yaw_mode = YawMode(False,0))
        start = time.time()
        return start, duration    

    def stop(self, duration):
        self.moveByVelocity(0, 0, 0, 1)
        start = time.time()
        return start, duration
    
    def yaw_right(self, duration):
        self.rotateByYawRate(30, duration)
        start = time.time()
        return start, duration
    
    def yaw_left(self, duration):
        self.rotateByYawRate(-30, duration)
        start = time.time()
        return start, duration
    
    
    def take_action(self, action):
		
        start = time.time()
        duration = 0 
        
        collided = False
		
        if action == 0:
            start, duration =  self.stop(1)
			
            while duration > time.time() - start:
                if self.getCollisionInfo().has_collided == True:
                    self.moveByVelocity(0, 0, 0, 1)
                    return True
            
        if action == 1:
            self.moveByVelocity(0, 0, 0, 1)
            start, duration = self.straight(1, 4)
        
            while duration > time.time() - start:
                if self.getCollisionInfo().has_collided == True:
                    self.moveByVelocity(0, 0, 0, 1)
                    return True    
            
        if action == 2:
            self.moveByVelocity(0, 0, 0, 1)
            start, duration = self.yaw_right(0.8)
            
            while duration > time.time() - start:
                if self.getCollisionInfo().has_collided == True:
                    self.moveByVelocity(0, 0, 0, 1)
                    return True
            
        if action == 3:
            self.moveByVelocity(0, 0, 0, 0.8)  
            start, duration = self.yaw_left(1)
            
            while duration > time.time() - start:
                if self.getCollisionInfo().has_collided == True:
                    self.moveByVelocity(0, 0, 0, 1)
                    return True
        
        return collided
    
    def goal_direction(self, goal, pos):
        
        pitch, roll, yaw  = self.getRollPitchYaw()
        yaw = math.degrees(yaw) 
        
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        direction = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(direction) - 180) % 360) - 180 
    
    def getScore(self, imgFull, hCenter, wCenter, size, checkMin):
        wsize2 = size /2
        hRange = range(int(hCenter-wsize2), int(hCenter+wsize2))
        wRange = range(int(wCenter-wsize2), int(wCenter+wsize2))
        
        sum = 0
        winMin = 9999999
        for i in hRange:
            for j in wRange:
                dist = imgFull[i,j]
                if checkMin:
                    winMin = min(dist, winMin)
                else:
                    winMin = dist
                sum += winMin
        result =sum/size/size
        return result
    
    def getSensorStates2(self, img, h, w, size):
        h2 = h/2
        w2 = w/2 
        offset = 50
        cscore = self.getScore(img, h2, w2, size, True)
        lscore = self.getScore(img, h2, w2-offset, size, True)
        rscore = self.getScore(img, h2, w2+offset, size, True)
        return [lscore, cscore, rscore]

    def getSensorStates(self):
        responses = self.simGetImages([ImageRequest(0, AirSimImageType.DepthPerspective, True)])
        response = responses[0]
        
    
        self.img1 = self.img2
        self.img2 = self.getPfmArray(response)
        img2 = self.img2
        result = [100.0, 100.0, 100.0]
        if len(img2) > 1:
            h = 144
            w = 256
            size = 20
            if self.img1 is not None and img2 is not None:
                result = self.getSensorStates2(img2, h, w, size)
        return result
    
    
    def getScreenDepthVis(self):

        #Get screenshot from the second monitotr and crop just the DepthVis
        for displayNumber, rect in enumerate(getDisplayRects(), 1):
            if displayNumber == 2:
                screen = getRectAsImage(rect)
        
        screen = np.array(screen)
        
        DepthVis = screen[762:1000, 98:525] # NOTE: its img[y: y + h, x: x + w]
        
        DepthVis_gray = cv2.cvtColor(DepthVis, cv2.COLOR_BGR2GRAY)
        
        small = cv2.resize(DepthVis_gray, (0,0), fx=0.5, fy=0.5) 
        
        #cv2.imshow("Test", DepthVis_gray)
        #cv2.waitKey(0)
        
        return DepthVis_gray

    def reset(self):
        
        client.reset()     # We have to use client here cause otherewise the progra, stuck in a endless loop. Dont know why . . 
        self.enableApiControl(True)
        self.armDisarm(True)
        self.moveToZ(self.z, 3) 
        time.sleep(2)
        
