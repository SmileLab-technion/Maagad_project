import tensorflow as tf
from keras.layers import Input, Dense
import tensorflow_probability as tfp
import numpy as np
from robosuite.controllers.logz import *
import time
import os
from robosuite.controllers.helping_func_class import LinearSchedule
import gym
from keras.models import load_model

class PPO(object):
    def __init__(
        self,
        Env,
        Render_env=None,
        sess=None,
        exp_name="try",
        n_iter=100,
        horizon=50,
        paths=50,
        reward_to_go=True,
        nn_baseline=True,
        gama=0.99,
        n_layers_policy=2,
        size_layer_policy=256,
        policy_activation=tf.tanh,
        policy_output_activation=None,
        n_layers_value=3,
        size_layers_value=512,
        value_activation = tf.tanh,
        value_output_activation=None,
        e_clip=0.2,
        Load_nn=False,
        learning_rate=5e-4,
        seed=1,
        critic=True,
        critic_loop=4,
        episodes=np.inf
    ):
        """"creating ppo object """
        #Env is the mujoco env child.
        self.env = Env

        #render env
        self.render_env=Render_env
        #Env observation space
        self.obs_space=self.env.observation_space.shape[0]
        # Env action space

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_space = self.env.action_space.shape[0]
        else:
            self.action_space = self.env.action_space.n

        # Tensor flow session.
        self.sess=sess
        #exp_name:
        self.exp_name=exp_name
        #log_dir
        self.logdir= exp_name + '_' + 'PPO' + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        self.logdir= os.path.join('data', self.logdir)
        self.robo=False
        if "Baxter" in self.env.__class__.__name__ or "Saywer" in self.env.__class__.__name__:
            self.robo=True
        # Number of iteration to run before backpropogation
        self.n_iter=n_iter
        #Number of paths to collect
        self.paths=paths
        #how far in the future to go
        self.horizon=horizon
        # Reward to go or not
        self.reward_to_go=reward_to_go
        # Use baseline to decrease variance
        self.nn_baseline=nn_baseline
        #discount factor:
        self.gama=gama
        # Number of layers policy
        self.n_layers_policy=n_layers_policy
        # number of neurons in a layer policy
        self.size_layer_policy= size_layer_policy
        # policy activation
        self.activation_policy = policy_activation
        # policy output activation
        self.output_activation_policy = policy_output_activation
        # Number of layers value
        self.n_layers_value = n_layers_value
        # number of neurons in a layer value
        self.size_layer_value= size_layers_value
        # value activation
        self.activation_value = value_activation
        # value output activation
        self.output_activation_value = value_output_activation
        # clipping the change in policy 0.2 value in the paper
        self.e_clip= e_clip
        #learning or Executing available policy
        self.load_nn=Load_nn
        #learning rate
        self.learning_rate_schedule =LinearSchedule(int(n_iter/4),learning_rate,learning_rate/10)
        self.learning_rate_schedule_value = LinearSchedule(int(n_iter / 4), learning_rate/10, learning_rate/10)
        #exploration schedul:
        self.exploration=LinearSchedule(int(n_iter/2), 0.1,1)
        #NN_saving_path:
        self.save_path = os.path.join('NN_models', str(self.exp_name))
        #critic
        self.critic=critic
        self.critic_loop=critic_loop
        self.episodes=episodes
        #seed tf
        tf.random.set_random_seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def _build_mlp(
            self,
            input_placeholder,
            output_size,
            scope
    ):
        nn_name = 'policy' if not scope.find('policy') == -1 else 'value'

        [n_layers,size_layer,activation,output_activation]=[self.__dict__["n_layers_"+str(nn_name)],self.__dict__["size_layer_"+str(nn_name)],
                                                            self.__dict__["activation_"+str(nn_name)],self.__dict__["output_activation_"+str(nn_name)]]

        with tf.variable_scope(scope):
            x = Input(tensor=input_placeholder)
            for i in range(n_layers):
                x = Dense(size_layer, activation=activation, name='fc' + str(i))(x)
            output_placeholder = Dense(output_size, activation=output_activation, name=str(nn_name) + "_action")(x)

        return output_placeholder


    def _locating_tf(self):

        [ob_dim,ac_dim]=[self.obs_space,self.action_space]

        self.sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)

        self.sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
        self.old_log_prob = tf.placeholder(shape=[None], name="old_prob", dtype=tf.float32)

            # Define a placeholder for advantages
        self.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        self.y_t = tf.placeholder(shape=[None], name="rew", dtype=tf.float32)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

    def _discount_rewards_to_go(self,rewards):
        res = []
        future_reward = 0
        gamma=self.gama

        for r in reversed(rewards):
            future_reward = future_reward * gamma + r
            res.append(future_reward)
        # return the ^-1 list nice way:)
        return res[::-1]

    #dont take into acount causality:
    def _sum_discount_rewards(self,rewards):
        gamma = self.gama
        return sum((gamma ** i) * rewards[i] for i in range(len(rewards)))


    def _next_step_policy(self):
            self.sy_mean = self._build_mlp(
                input_placeholder=self.sy_ob_no,
                output_size=self.action_space,
                scope="policy_nn")

            # logstd should just be a trainable variable, not a network output??
            self.sy_logstd = tf.get_variable("logstd".format(np.random.rand(1)[0]), shape=[self.action_space])

            # random_normal just give me a number between -1 to 1. if we multiply this number by sigma we like sampling from
            # The normal distribution. sample=Mu+sigma*Z,Z~N(0,1)

            self.sy_sampled_ac = tf.math.add(self.sy_mean , tf.multiply(tf.exp(self.sy_logstd),tf.random_normal(tf.shape(self.sy_mean))),name="sampled_action")

            # Hint: Use the log probability under a multivariate gaussian.
            dist = tfp.distributions.MultivariateNormalDiag(loc=self.sy_mean, scale_diag=tf.exp(self.sy_logstd),name="myOutput")
            # caculate -log (PI(a))-we just need to enter a from our sampling along the paths.
            self.sy_logprob_n = -dist.log_prob(self.sy_ac_na)


    def _next_step_value(self):

        self.baseline_prediction = tf.squeeze(self._build_mlp(
            input_placeholder=self.sy_ob_no,
            output_size=1,
            scope="value_nn"))
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        self.baseline_target = tf.placeholder(shape=[None], dtype=tf.float32,name="baseline_target")

    def initiate_ppo_controler(self):

        self._locating_tf()
        self._next_step_policy()
        self._next_step_value()


        self.baseline_loss = tf.losses.mean_squared_error(predictions=self.baseline_prediction, labels=self.baseline_target)

        #self.baseline_update_op=tf.get_default_graph.__dict__
        self.baseline_update_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.baseline_loss)

        self.entropy_loss = -tf.reduce_sum(tf.multiply(tf.sigmoid(self.sy_sampled_ac), tf.log(tf.sigmoid(self.sy_sampled_ac))))
        self.critic_loss =tf.losses.mean_squared_error(predictions=self.baseline_prediction, labels=self.y_t)
        self.update_critic_loss = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.critic_loss)
        self.prob_ratio = tf.exp(self.sy_logprob_n - self.old_log_prob)
        self.clip_prob = tf.clip_by_value(self.prob_ratio, 1. - self.e_clip, 1. + self.e_clip)
        self.weighted_negative_likelihood = tf.multiply(self.sy_logprob_n, self.sy_adv_n)
        self.ppo_loss=tf.reduce_mean(tf.minimum(tf.multiply(self.prob_ratio,self.sy_adv_n), tf.multiply(self.clip_prob, self.sy_adv_n)))
        self.update_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.ppo_loss)

        self._initiate_session()
        # if not self.load_nn:
        #     self.sess.run(tf.global_variables_initializer())
        #
        # if self.load_nn:
        #
        #     self.sess.run(tf.global_variables_initializer())
        #     path = os.path.join('NN_models', str(self.exp_name)+".meta")
        #     #path_meta = os.path.join(path, str(self.exp_name))
        #     saver = tf.train.import_meta_graph(path)
        #     saver.restore(self.sess,tf.train.latest_checkpoint('NN_models'))
        #
        #     tvars = tf.trainable_variables()
        #     tvars_vals = self.sess.run(tvars)
        #     File_object2 = open("load.txt", "w")
        #     for var, val in zip(tvars, tvars_vals):
        #         File_object2.writelines(str(var.name) + "<---------->" + str(val))
        #     File_object2.close()


    def _initiate_session(self):

        if not self.sess :
            tf_config = tf.ConfigProto(device_count={"CPU": 8})
            self.sess = tf.Session(config=tf_config)
            #self.sess.__enter__()
            self.sess.run(tf.global_variables_initializer())


    def _V_future(self,V_s,steps):
        for i in steps:
            V_s[i - 1] = 0
        V_s = np.array(np.append(V_s, 0))
        return V_s[1:]

    def _pathlength(self,path):
        return len(path["reward"])

    def train(self,load_all=False):
        configure_output_dir(self.logdir)
        start = time.time()
        A = np.ones((1, self.obs_space))
        render=False
        if os.path.isfile(self.save_path + ".meta"):
            self.load_tf_model()
            mean=self.sess.run(self.baseline_prediction,feed_dict={self.sy_ob_no: A})
            print(mean)
        for itr in range(self.n_iter):

            if itr==0:
                self.env = self.render_env
                render=True

            print("********** Iteration %i ************ " % itr)
            paths=[]


            for num_path in range(self.paths):
                ob = self.env.reset()
                if self.robo:
                    ob=np.concatenate((ob['robot-state'],ob['object-state']),axis=-1)
                obs,acs,rewards,adv_n,yt=[],[],[],[],[]
                steps = 0
                for t_horizon in range(self.horizon):
                    obs.append(ob)
                    #A = self.env.sim.data.body_xmat[self.env.cyl_body_id]
                    #B=self.env.sim.data.body_xpos[self.env.cyl_body_id]
                    #C=self.env.sim.data.efc_J
                    target_jacp = np.zeros(3 * self.env.sim.model.nv)

                    ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob[None, :]})
                    value= self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob[None, :]})
                    if self.exploration.value(1000)>np.random.rand(1)[0]:
                        ac =np.expand_dims((np.random.randn(self.action_space)),axis=0)

                    #ac[0, 7:]=0
                    if render:
                        self.env.render()
                        time.sleep(0.1)

                    acs.append(ac[0,:])
                    #R=self.env.sim.data.get_body_jacp('world',jacp=target_jacp)
                    ob, rew, done, _ = self.env.step(ac[0,:])
                    #N=self.env.sim.forward()
                    if self.robo:
                        ob=np.concatenate((ob['robot-state'],ob['object-state']),axis=-1)
                    rewards.append(rew)
                    steps+=1
                    if done:
                        adv_n.append(rew - value)
                        yt.append(rew)
                        break
                    next_value=self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob[None, :]})
                    adv_n.append(rew+self.gama*next_value-value)
                    yt.append(rew+self.gama*next_value)

                path = {"observation": np.array(obs),
                        "reward": np.array(rewards),
                        "action": np.array(acs),
                        "steps": steps,
                        "adv_n":adv_n,
                        "yt":yt}

                paths.append(path)

            steps = [path["steps"] for path in paths]
            ob_no = np.concatenate([path["observation"] for path in paths])
            ac_na = np.concatenate([path["action"] for path in paths])
            r_s = np.concatenate([path["reward"] for path in paths])
            adv_n=np.concatenate([path["adv_n"] for path in paths])
            y_t = np.concatenate([path["yt"] for path in paths])

            log_old_temp = self.sess.run(self.sy_logprob_n, feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na})
            q_n=[]

            if self.critic:
                #V_s = self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob_no})
                #V_s_1 = self._V_future(V_s, steps)
                #y_t = r_s + self.gama * V_s_1
                #adv_n = y_t - V_s

                for i in range(self.critic_loop):
                    _, loss_critic = self.sess.run([self.update_critic_loss, self.critic_loss],
                                                  feed_dict={self.sy_ob_no: ob_no,self.y_t:y_t, self.learning_rate: self.learning_rate_schedule_value.value(itr)})


            else:
                loss_critic = None

                if self.reward_to_go:
                    q_n = np.concatenate([self._discount_rewards_to_go(path["reward"]) for path in paths])
                else:
                    q_n = np.concatenate(
                        [[self._sum_discount_rewards(path["reward"])] * self._pathlength(path) for path in paths])

                if self.nn_baseline and itr > 0:
                    # ------------------------Caculate Advantage in baseline mode-----------------------
                    b_n = self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob_no})
                    b_n = (b_n - np.mean(b_n)) / np.std(b_n)
                    b_n = np.mean(q_n) + b_n * np.std(q_n)
                    adv_n = q_n - b_n
                    # -----------------------------------Update baseline NN-----------------------------------
                    scaled_q = (q_n - np.mean(q_n)) / np.std(q_n)
                    self.sess.run(self.baseline_update_op, feed_dict={self.sy_ob_no: ob_no, self.baseline_target: scaled_q,self.learning_rate:self.learning_rate_schedule.value(itr)})
                else:
                    adv_n = q_n.copy()

            #q_n = np.concatenate([path["reward"] for path in paths])
            #V_n = self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob_no})
            #V_n = (V_n - np.mean(V_n)) / np.std(V_n)
            #V_n_1 = self._V_future(V_n, steps)
            #y_t = q_n + self.gama * V_n_1
            #adv_n = y_t - V_n

            _, loss_value = self.sess.run([self.update_op, self.ppo_loss],
                                     feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_adv_n: adv_n,
                                                self.old_log_prob: log_old_temp, self.learning_rate:self.learning_rate_schedule.value(itr)})

            returns = [path["reward"].sum() for path in paths]
            ep_lengths = [self._pathlength(path) for path in paths]
            log_tabular("Time", time.time() - start)
            log_tabular("Iteration", itr)
            log_tabular("AverageReturn", np.mean(returns))
            log_tabular("StdReturn", np.std(returns))
            log_tabular("MaxReturn", np.max(returns))
            log_tabular("MinReturn", np.min(returns))
            log_tabular("EpLenMean", np.mean(ep_lengths))
            log_tabular("EpLenStd", np.std(ep_lengths))
            log_tabular("Loss", loss_value)
            log_tabular("Loss_critic",loss_critic)
            log_tabular("Learning_rate", self.learning_rate_schedule.value(itr))
            log_tabular("Exploration",self.exploration.value(itr))
            dump_tabular()


        if not os.path.isfile(self.save_path + ".meta"):
            saver = tf.train.Saver(save_relative_paths=True)

            saver.save(self.sess, self.save_path)
            print("saved")
            mean=self.sess.run(self.baseline_prediction,feed_dict={self.sy_ob_no: A})
            print(mean)
        else:
            saver = tf.train.Saver(save_relative_paths=True)

            saver.save(self.sess, self.save_path+str(int(np.random.rand(1)[0]*10)))
            print("saved")
            mean = self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: A})
            print(mean)

        self.finish_sess()

            #pickle_tf_vars()



    def render(self, env, n_iter=100):

        rewards=[]
        if env.has_renderer == False:
            raise ValueError("require has_renderer=True")
        else:
            ob = self.env.reset()
            ob = np.concatenate((ob['robot-state'],ob['object-state']),axis=-1)
            for iter in range(n_iter):
                ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob[None, :]})
                ac[0, 7:] = 0
                env.render()
                time.sleep(0.1)
                ob, rew, done, _ = env.step(ac[0, :])
                rewards.append(rew)
                ob = np.concatenate((ob['robot-state'], ob['object-state']), axis=-1)
                if done:
                    break
            print(np.sum(rewards))

    def finish_sess(self):
        self.sess.close()

    def save_tf_model(self):
        path = os.path.join('NN_models',str(self.exp_name))
        saver= tf.train.Saver(save_relative_paths=True)
        #temp=os.path.join('nn_weights', self.logdir)
        #tf.saved_model.simple_save(self.sess,temp,inputs={"ob":self.sy_ob_no},outputs={"myOutput":self.sy_logprob_n,"baseline_target":self.baseline_prediction})
        #saver.save(self.sess,path)
        save_path = saver.save(self.sess, path)
        print("Model saved in path: %s" % save_path)


    def load_tf_model(self):
        #path = os.path.join('NN_models', str(self.exp_name) + ".meta")
        #self.sess=tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        #new_saver = tf.train.import_meta_graph(path)
        #new_saver.restore(self.sess, tf.train.latest_checkpoint('NN_models/.'))



        #self.sess.run(tf.global_variables_initializer())
        #graph = tf.get_default_graph()
        #self.sy_sampled_ac= graph.get_tensor_by_name("sampled_action:0")
        #self.sy_ob_no =graph.get_tensor_by_name("ob:0")
        #self.sy_ac_na = graph.get_tensor_by_name("ac:0")
        #dist = graph.get_tensor_by_name("myOutput:0")
        #self.sy_logprob_n=-dist.log_prob(self.sy_ac_na)

        #tvars = tf.trainable_variables()
        #tvars_vals = self.sess.run(tvars)

        #File_object2 = open("load.txt", "w")
        #for var, val in zip(tvars, tvars_vals):
        #    File_object2.writelines(str(var.name) + "<---------->" + str(val))
        #File_object2.close()
        #tf.reset_default_graph()
        #self.sess.run(tf.global_variables_initializer())
        path = os.path.join('NN_models', str(self.exp_name))
        saver = tf.train.Saver(save_relative_paths=True)
        saver.restore(self.sess, save_path=path)
        # new_saver = tf.train.import_meta_graph(path)
        # new_saver.restore(self.sess, tf.train.latest_checkpoint('NN_models/.'))
        # #saver = tf.train.Saver(save_relative_paths=True)
        #
        # #save_path = saver.restore(self.sess, "NN_models/model.ckpt")
        #
        # tvars = tf.all_variables()
        # tvars_vals = self.sess.run(tvars)
        # File_object2 = open("load.txt", "w")
        # for var, val in zip(tvars, tvars_vals):
        #     File_object2.writelines(str(var.name) + "<---------->" + str(val))
        # File_object2.close()
        #
        # graph = tf.get_default_graph()
        # self.sess.run(tf.global_variables_initializer())
        # self.sy_ac_na = graph.get_tensor_by_name("ac:0")
        # self.sy_ob_no = graph.get_tensor_by_name("ob:0")
        # self.sy_sampled_ac =graph.get_tensor_by_name("sampled_action:0")
        # self.sy_logprob_n = graph.get_tensor_by_name("Neg:0")
        # self.baseline_prediction=graph.get_tensor_by_name("Squeeze:0")
        # self.baseline_loss =graph.get_tensor_by_name("mean_squared_error/value:0")
        #
        # self.baseline_update_op =graph.get_tensor_by_name('value_nn/fc0/kernel/Adam:0')
        # #self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.baseline_loss)
        # self.entropy_loss =graph.get_tensor_by_name("Neg_1:0")
        # self.critic_loss =graph.get_tensor_by_name("Mean:0")
        # self.weighted_negative_likelihood =graph.get_tensor_by_name("Mul_2:0")
        # self.ppo_loss =graph.get_tensor_by_name("Mean_1:0")
        # self.update_op =graph.get_tensor_by_name("policy_nn/fc0/kernel/Adam:0")
        #
        # self.prob_ratio = graph.get_tensor_by_name("Exp_2:0")
        # self.clip_prob = graph.get_tensor_by_name("clip_by_value:0")
        #
        #
        # self.old_log_prob = graph.get_tensor_by_name("old_prob:0")
        #
        #     # Define a placeholder for advantages
        # self.sy_adv_n =graph.get_tensor_by_name("adv:0")
        # self.reward = graph.get_tensor_by_name("rew:0")
        # self.mean=graph.get_tensor_by_name("policy_nn/policy_action/BiasAdd:0")
        # self.sy_logstd=graph.get_tensor_by_name("logstd:0")
        # self.baseline_target=graph.get_tensor_by_name('baseline_target:0')
        #init = tf.global_variables_initializer()
        #dist = graph.get_tensor_by_name("myOutput:0")
        #self.sy_logprob_n = -dist.log_prob(self.sy_ac_na)
        #temp=os.path.join('nn_weights', self.logdir)
        # tf.saved_model.simple_save(self.sess,temp,inputs={"ob":self.sy_ob_no},outputs={"myOutput":self.sy_logprob_n,"baseline_target":self.baseline_prediction})
        # saver.save(self.sess,path)

        print("Model restored.")






































