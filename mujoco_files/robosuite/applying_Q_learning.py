import numpy as np
import robosuite as suite
import time
from robosuite.controllers import Double_Q
import gym

env_name='CartPole-v1'
#env_name='PongNoFrameskip-v4'
env=gym.make(env_name)
env=gym.wrappers.Monitor(env,'monitor',video_callable=lambda x: False,force=True)
A=Double_Q.QLearner(env=env,exp_name="try1")
A.learn(1000000)

