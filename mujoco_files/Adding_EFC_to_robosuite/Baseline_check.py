import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from robosuite.wrappers import GymWrapper
import robosuite as suite
import numpy as np
import time


def evaluate(model,env, num_steps=1000,render=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)
        if render:
            env.render()
            time.sleep(0.1)
        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    #print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward

if __name__ == "__main__":
    env = GymWrapper(suite.make("BaxterPegInHole", has_renderer=True, reward_shaping=True,control_freq=100,action_space_def="EOF"))

    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    #env = gym.make('CartPole-v1')
    #env = DummyVecEnv([lambda: env])

    #model = PPO2(MlpPolicy, env, verbose=0)
    #model.learn(total_timesteps=100000)
    #model.save("PPO2_EOF")
    #del model
    #env.close()
    #model = PPO2(MlpPolicy, env, verbose=3)
    model = PPO2.load("PPO_EOF_3")
    model.env=env
    model.learn(total_timesteps=200000)
    mean_reward=0
    for i in range(40):
        mean_reward += evaluate(model, env=env, num_steps=200,render=False)
    print(mean_reward/40)
    #
    model.save("PPO_EOF_3")
    #mean_reward=evaluate(model,env=env,num_steps=1000)

    del model
    env.close()