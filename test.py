import realant_sim
import gym
import time
import numpy as np
import math

class HexaControllerSine () :

    def __init__(self, params=None, array_dim=100):
        self.array_dim = array_dim
        self._params = None

    def nextCommand(self, t):
        # Control parameters
        # steer = 0.0 #Move in different directions
        # step_size = 1.0 # Walk with different step_size forward or backward
        # leg_extension = 1.0 #Walk on different terrain
        # leg_extension_offset = -1.0

        steer = self._params[0] #Move in different directions
        step_size = self._params[1] # Walk with different step_size forward or backward
        leg_extension = self._params[2] #Walk on different terrain
        leg_extension_offset = self._params[3]

        # Robot specific parameters
        swing_limit = 0.5
        extension_limit = 0.4
        speed = 10
        
        A = np.clip(step_size + steer, -1, 1)
        B = np.clip(step_size - steer, -1, 1)
        extension = extension_limit * (leg_extension+1.0) * 0.5
        max_extension = np.clip(extension + extension_limit*leg_extension_offset, 0, extension)
        min_extension = np.clip(-extension + extension_limit*leg_extension_offset, -extension, 0)

        #We want legs to move sinusoidally, smoothly
        fl = math.sin(t * speed) * (swing_limit * A)
        br = math.sin(t * speed) * (swing_limit * B)
        fr = math.sin(t * speed + math.pi) * (swing_limit * B)
        bl = math.sin(t * speed + math.pi) * (swing_limit * A)
        
        #We can legs to reach extreme extension as quickly as possible: More like a smoothed square wave
        e1 = np.clip(3.0 * math.sin(t * speed + math.pi/2), min_extension, max_extension)
        e2 = np.clip(3.0 * math.sin(t * speed + math.pi + math.pi/2), min_extension, max_extension)
        # return np.array([bl,e1,e1, 0,0,0, fl,e2,e2, -fr,e1,e1, 0,0,0, -br,e2,e2]) * np.pi/4
        return np.array([fl,e2, -fr,e1, bl,e1, -br,e2]) #* np.pi/4
        
        #Swing
        #0: back_left
        #6: front_left
        #9: front_right
        #15: back_right

    def setParams(self, params, array_dim=100):
        self._params = params

    def setRandom(self):
        self._random = True
        self.setParams(np.random.rand(4) * 2.0 - 1.0)

    def getParams(self):  
        return self._params


env = gym.make('RealAntBullet-v0', render=True)

state = env.reset()

ctlr = HexaControllerSine()
ctlr.setParams(np.array([0, 1., 1., 0.]))

hip =  [1, -1, -1, 1] + [1, -1, -1, 1]
knee = [-1, -1, 0, 0] + [-1, -1, 0, 0]

id = np.random.randint(0,3)
ext = np.random.randn()*2-1
fl_h = np.array(hip[id:id+4]) * ext
fl_k = np.array(knee[id:id+4])

id = np.random.randint(0,3)
ext = np.random.randn()*2-1
fr_h = np.array(hip[id:id+4]) * ext
fr_k = np.array(knee[id:id+4])

id = np.random.randint(0,3)
ext = np.random.randn()*2-1
bl_h = np.array(hip[id:id+4]) * ext
bl_k = np.array(knee[id:id+4])

id = np.random.randint(0,3)
ext = np.random.randn()*2-1
br_h = np.array(hip[id:id+4]) * ext
br_k = np.array(knee[id:id+4])

act_mat = np.array([fl_h, fl_k, fr_h, fr_k, bl_h, bl_k, br_h, br_k]).transpose()

for i in range(1000):

    act = act_mat[0,:]
    for k in range(100): env.step(act)

    act = act_mat[1,:]
    for k in range(100): env.step(act)

    act = act_mat[2,:]
    for k in range(100): env.step(act)
    
    act = act_mat[3,:]
    for k in range(100): env.step(act)

    time.sleep(0.1)