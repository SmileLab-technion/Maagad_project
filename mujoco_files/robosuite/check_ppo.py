import gym
from robosuite.controllers import PPO

env_name = 'InvertedPendulum-v2'
env = gym.make(env_name)

PPO_control=PPO(Env=env,n_iter=100,paths=20,horizon=100,Load_nn=False,exp_name="Ant_1",critic=True)
PPO_control.initiate_ppo_controler()
PPO_control.train()