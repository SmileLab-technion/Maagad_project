import tensorflow as tf
from keras.layers import Input, Dense
import tensorflow_probability as tfp
import numpy as np
from robosuite.controllers.logz import *
import time
import os
import tensorflow.contrib.layers as layers
from collections import namedtuple
from robosuite.controllers.helping_func_class import *
from keras.models import load_model
import gym
import sys
from gym import wrappers

class QLearner(object):
    def __init__(
        self,
        env,
        sess=None,
        optimizer_spec=None,
        exp_name="try",
        optimization_method=tf.train.AdamOptimizer,
        exploration=LinearSchedule(10000, 0.1),
        stopping_criterion=None,
        replay_buffer_size=1000,
        batch_size=32,
        paths=50,
        learning_starts=500,
        learning_freq=4,
        frame_history_len=1,
        target_update_freq=50,
        num_timesteps=int(1e6),
        grad_norm_clipping=10,
        gamma=1,
        rew_file=None,
        double_q=True,
        lander=False,
        Q_input_vision=True,
        n_layers_Q=2,
        size_layer_Q=64,
        Q_network_type='VGG16',#['VGG16', 'RESNET52', 'InceptionV3', 'Xception', 'DenseNet169', 'DenseNet121'],
        IMG_SIZE=(84,84),
        Q_activation=tf.tanh,
        Q_output_activation=None,
        Load_nn=False,
        learning_rate=5e-4
    ):
        # Env is the mujoco env child.
        self.env = env
        # continues/discrete action space
        self.env_type=True if "gym" in self.env.spec._entry_point else False

        if self.env_type:
            self.disrete=isinstance(self.env.action_space, gym.spaces.Discrete)
            self.obs_space = self.env.observation_space.shape
            self.action_space = self.env.action_space.n if self.disrete else self.env.action_space.shape[0]
        else:
            self.disrete =False
            self.obs_space = self.env.observation_space
            self.action_space = self.env.dof

        # Tensor flow session.
        self.sess = get_session()
        # exp_name:
        self.exp_name = exp_name
        # log_dir
        self.logdir = exp_name + '_' + 'Double_Q' + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        self.logdir = os.path.join('data', self.logdir)
        self.gamma=gamma
        self.num_timesteps=num_timesteps
        self.optimization_method=optimizer_defenition(self.num_timesteps,optimization_method)
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.learning_starts = learning_starts
        self.stopping_criterion = stopping_criterion
        self.exploration = exploration

        #if the shape is only 1D then it's not an image input:
        if len(self.obs_space) == 1:
            input_shape = self.obs_space
            self.q_func=self.q_func_build_not_img

        else:
            img_h, img_w, img_c = self.obs_space.shape
            input_shape = (img_h, img_w, frame_history_len * img_c)
            self.q_func = self.q_func_build_img

        self.obs_t_ph = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(input_shape))
        # placeholder for current action
        if self.disrete:
            self.act_t_ph = tf.placeholder(tf.int32, [None])
        else:
            self.act_t_ph = tf.placeholder(tf.float32, [None]+list(self.action_space))
        # placeholder for current reward
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        # placeholder for next observation (or state)
        self.obs_tp1_ph = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(input_shape))
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        # casting to float on GPU ensures lower data transfer times.
        if lander:
            obs_t_float = self.obs_t_ph
            obs_tp1_float = self.obs_tp1_ph
        else:
            obs_t_float = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
            obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0


        #-------------------------------creating q networks---------------------------------------------------------

        """
        remember yi=r(si,ai)+gamma*max ai'(Q(si',ai')) || yi is the approximation of Q(si,ai) according to bellman equation
        phi=argmin phi ( Q(s,a)-yi)^2)
        
        implementation methods:
        1) create 2 diffrent network Q(s,a) and Q(s',a'). 
        2)take Q(s',a') and reduce_max axis=1 which is taking the max a'(Q(s',a'))=self.target_action
        3)caculate yi=r(si,ai)+gamma*self.target_action
        4)caculate Q(si,ai), we created placeholder self.act_t_ph for choosed action input, Q(si,ai) is q_val*tf.one_hot(self.act_t_ph, self.action_space)
        which create on hot vector on the choosed action. doing that + reduce_sum elimante all unchosed actions.
        5)using huber loss which combines the benefit of the L1 and L2 loss. L2 when close to convergese and L1 when far.
        6)change phi<-argmax phi(|yi-Q(si,ai)|)
        7)once in 10,000 steps Q(si',ai') parameters= Q(si,ai), this once loop change yi slowly give Q time to converge to yi.
        """
        q_val = self.q_func(obs_t_float, self.action_space, scope="q_func")
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_func")
        self.action = tf.argmax(q_val, axis=1)

        target_q_val =self.q_func(obs_tp1_float, self.action_space, scope="target_q_func")
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_q_func")

        if double_q:
            online_q_val = self.q_func(obs_tp1_float, self.action_space, scope="q_func", reuse=True)
            online_action = tf.argmax(online_q_val, axis=1)
            self.target_action = tf.reduce_sum(target_q_val * tf.one_hot(online_action, self.action_space), axis=1)
        else:
            self.target_action = tf.reduce_max(target_q_val, axis=1)

        y = self.rew_t_ph + (1. - self.done_mask_ph) * self.gamma * self.target_action
        q_s_a_current = tf.reduce_sum(tf.one_hot(self.act_t_ph, self.action_space) * q_val, axis=1)

        self.total_error = tf.reduce_mean(huber_loss(q_s_a_current - y))

        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")

        optimizer = self.optimization_method.constructor(learning_rate=self.learning_rate, **self.optimization_method.kwargs)

        #check gradients to calc total error and check if they are lower then 10
        self.train_fn = minimize_and_clip(optimizer, self.total_error,
                                          var_list=q_func_vars, clip_val=grad_norm_clipping)


        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

        # construct the replay buffer
        self.replay_buffer = ReplayBuffer(
            replay_buffer_size, frame_history_len, lander=lander)
        self.replay_buffer_idx = None
        self.model_initialized = False
        self.num_param_updates = 0
        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.last_obs = self.env.reset()
        self.log_every_n_steps = 10000

        self.start_time = None
        self.t = 0



    def q_func_build_img(self,img_in, num_actions, scope, reuse=False):
        # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        with tf.variable_scope(scope, reuse=reuse):
            out = img_in
            with tf.variable_scope("convnet"):
                # original architecture
                out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            out = layers.flatten(out)
            with tf.variable_scope("action_value"):
                out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

            return out


    def q_func_build_not_img(self,ram_in, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            out = ram_in
            # out = tf.concat(1,(ram_in[:,4:5],ram_in[:,8:9],ram_in[:,11:13],ram_in[:,21:22],ram_in[:,50:51], ram_in[:,60:61],ram_in[:,64:65]))
            with tf.variable_scope("action_value"):
                #out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

            return out

    def stopping_criterion_met(self):
        return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

    def step_env(self):
        """"
        remember:
        one of the biggest problems in Q learning is the fact that the data Q(s,a) are highly correleted
        in time which means that in folowing some trajectory the Q(s,a) the i get in the k time steps will
        be just points a crose small part of the trajectory and not i.i.d points like should be in SGD.

        solution:
        because Q learning is of policy every data that we collect no metter with which policy is good for
        propogation. This is why we are going to use a buffer of 1M (observation,action,reward) for training
        then we will pull randomly from this pool.
        """

        #pop last observation
        idx = self.replay_buffer.store_frame(self.last_obs)
        recent_obs_encode = self.replay_buffer.encode_recent_observation()
        recent_obs = np.expand_dims(recent_obs_encode, 0)

        # if no initialized yet or random number<exploration take random action
        if not self.model_initialized or random.random() < self.exploration.value(self.t):
            action = self.env.action_space.sample()
        else:
            action = self.sess.run(self.action, feed_dict={self.obs_t_ph: recent_obs})[0]

        #step env:

        obs, reward, done, info = self.env.step(action)

        # if reach an episode boundary, get a new observation
        if done:
            obs = self.env.reset()

        #store observation action and reward:


        self.replay_buffer.store_effect(idx, action, reward, done)
        self.last_obs = obs

    def update_model(self):

        # K=4 every 4 steps we update the gradients.
        if (self.t > self.learning_starts and
            self.t % self.learning_freq == 0 and
                self.replay_buffer.can_sample(self.batch_size)):

            #sample batch size from buffer

            s_batch, a_batch, r_batch, sp_batch, done_mask_batch = self.replay_buffer.sample(
                self.batch_size)

            #if Model is not initialized:
            if not self.model_initialized:
                initialize_interdependent_variables(self.sess, tf.global_variables(), {
                                                    self.obs_t_ph: s_batch, self.obs_tp1_ph: sp_batch, })
                self.model_initialized = True

            #update gradients:
            feed_dict = {self.obs_t_ph:  s_batch,
                         self.act_t_ph: a_batch,
                         self.rew_t_ph: r_batch,
                         self.obs_tp1_ph: sp_batch,
                         self.done_mask_ph: done_mask_batch,
                         self.learning_rate: self.optimization_method.lr_schedule.value(self.t)}
            self.sess.run(self.train_fn, feed_dict=feed_dict)


            self.num_param_updates += 1

            #once in 10,000 times we will update Q(s',a') so yi doesn't change fast.
            if self.num_param_updates % self.target_update_freq == 0:
                self.sess.run(self.update_target_fn)

        self.t += 1

    def log_progress(self):
      episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

      if len(episode_rewards) > 0:
        self.mean_episode_reward = np.mean(episode_rewards[-100:])

      if len(episode_rewards) > 100:
        self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

      if self.t % self.log_every_n_steps == 0 and self.model_initialized:
        print("Timestep %d" % (self.t,))
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)
        print("episodes %d" % len(episode_rewards))
        print("exploration %f" % self.exploration.value(self.t))
        print("learning_rate %f" % self.optimization_method.lr_schedule.value(self.t))
        if self.start_time is not None:
          print("running time %f" % ((time.time() - self.start_time) / 60.))

        self.start_time = time.time()

        sys.stdout.flush()

        with open(self.logdir , 'wb') as f:
          pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)


    def learn(self,steps):
        for i in range(steps):
            if self.stopping_criterion_met():
                break

            self.step_env()
            # at this point, the environment should have been advanced one step (and
            # reset if done was true), and self.last_obs should point to the new latest
            # observation
            self.update_model()
            self.log_progress()




