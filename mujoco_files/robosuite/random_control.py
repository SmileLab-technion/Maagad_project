import numpy as np
import robosuite as suite
import time
# create environment instance
env = suite.make("BaxterPegInHole", has_renderer=True)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.dof)  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()
    time.sleep(0.1)