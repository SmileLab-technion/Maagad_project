import numpy as np
from mountain_car_with_data_collection import MountainCarWithResetEnv
import matplotlib.pyplot as plt
import os
import time
import os.path as osp

class ReplayBuffer(object):
    def __init__(self, size):

        self.size=size
        self.state=[]
        self.action=[]
        self.reward=[]
        self.done=[]
        self.s_tag=[]
        self.last_idx=0

    def push(self,states,actions,rewards,dones,s_tags):
        num=len(actions)
        pop_num=(self.last_idx+num)-self.size
        self.pop_begin(pop_num)

        for iter in range(len(actions)):
            self.state.append(states[iter])
            self.action.append(actions[iter])
            self.reward.append(rewards[iter])
            self.done.append(dones[iter])
            self.s_tag.append(s_tags[iter])

        self.last_idx+=num

    def pop_begin(self,n):
        if n<0:
            pass
        else:

            self.state=self.state[n:]
            self.action = self.action[n:]
            self.reward = self.reward[n:]
            self.done = self.done[n:]
            self.s_tag = self.s_tag[n:]
            self.last_idx+=-n

    def pop(self, n):
        if n>self.last_idx:
            n=self.last_idx

        states, actions, rewards, s_tags, done = [], [], [], [], []
        order= list(np.random.permutation(self.last_idx)[:n])
        order.sort(reverse=True)

        for i in order:
            states.append(self.state.pop(i))
            actions.append(self.action.pop(i))
            rewards.append(self.reward.pop(i))
            done.append(self.done.pop(i))
            s_tags.append(self.s_tag.pop(i))

        self.last_idx += -n

        return [states,actions,rewards,done,s_tags]



class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)



class QLearner(object):



    def __init__(
    self,
    env,
    exploration=LinearSchedule(100000, 0.1,1),
    stopping_criterion=None,
    replay_buffer_size=100000,
    batch_size=128,
    gamma=0.999,
    learning_starts=10000,
    learning_freq=4,
    grad_norm_clipping=4,
    grid_num=9,
    learing_rate=5e-2,
    seed=2):


        self.env = env
        self.action_space=env.action_space.n
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.stopping_criterion = stopping_criterion
        self.exploration = exploration
        self.gamma = gamma
        self.grad_clipping=grad_norm_clipping
        self.replay_buffer=ReplayBuffer(replay_buffer_size)
        self.t=0
        self.seed = seed
        np.random.seed(seed)
        self.state_min_bound = np.array((env.min_position, -env.max_speed))
        self.state_max_bound = np.array((env.max_position, env.max_speed))
        self.state_space=int(grid_num)+1
        self.w=np.random.randn(self.action_space * self.state_space)

        self.grider,self.rbf_sigma=self.get_grid(grid_num)
        self.learning_rate=learing_rate
        self.sample(learning_starts)
        self.w_save=[]
        self.w_save.append(np.sum(self.w))


    def normalization(self,sample):
        d_mean = np.array([-2.98842473e-01, -6.65825984e-05])
        d_std = np.array([0.52020299, 0.040452])

        return (sample-d_mean)/(d_std)

    def grid(self,grid_num,max=1):

        num = max / (grid_num - 1)
        x = np.arange(0- 1e-8, max + 1e-8, num)
        y = np.arange(0 - 1e-8, max + 1e-8, num)
        sigma = x[1] - x[0]
        xx, yy = np.meshgrid(x, y, sparse=False)
        xx = np.reshape(xx, (xx.shape[0] * xx.shape[1], 1))
        yy = np.reshape(yy, (yy.shape[0] * yy.shape[1], 1))
        grider = np.concatenate((xx, yy), axis=1)

        return grider, sigma

    def get_grid(self, nFeatures=4):
        assert np.sqrt(nFeatures) == int(np.sqrt(nFeatures))

        grid_size = int(np.log2(nFeatures))

        s = self.env.min_position + (self.env.max_position - self.env.min_position) / (grid_size + 1)
        pos_grid = np.linspace(s, self.env.max_position, grid_size, endpoint=False)

        s = -self.env.max_speed + (self.env.max_speed - -self.env.max_speed) / (grid_size + 1)
        vel_grid = np.linspace(s, self.env.max_speed, grid_size, endpoint=False)
        centers = np.zeros((nFeatures, 2))
        idx = 0

        for i in range(grid_size):
            for j in range(grid_size):
                centers[idx, :] = [pos_grid[i], vel_grid[j]]
                idx += 1

        return centers, 1


    def rbf_calc(self, sample):
        all_states = []
        rbf_sigma = 2 * self.rbf_sigma ** 2
        for mean in self.grider:

            new_state = np.exp((-np.linalg.norm(sample - mean, 2, axis=-1) ** 2) / rbf_sigma)
            all_states.append(new_state)
        new_state = (new_state - new_state + 1)
        all_states.append(new_state)
        return np.array(all_states).T


    def create_phi(self,state, action, done):
        n_iter=self.batch_size if len(np.array(state).shape)>1 else 1
        phi = np.zeros((n_iter,self.state_space * self.action_space))
        if n_iter==1:
            phi[0, action[0] * self.state_space:(action[0] + 1) * self.state_space]=state *(1 - done[0])
        else:
            for i in range(n_iter):
                phi[i,action[i] * self.state_space:(action[i] + 1) * self.state_space] =state[i] *(1 - done[i])
        return phi


    def calc_Q(self,phi):

        return np.dot(phi, self.w)



    def greedy_policy(self,state,done):
        size=self.batch_size if len(np.array(state).shape)>1 else 1
        Q=np.zeros((self.action_space,size))
        phi_0 = self.create_phi(state, [0]*size, done)
        phi_1=self.create_phi(state, [1]*size, done)
        phi_2=self.create_phi(state, [2]*size, done)
        phi_state = np.array([phi_0, phi_1, phi_2])

        for i in range(self.action_space):
            Q[i,:]=self.calc_Q(phi_state[i,:,:])


        return np.argmax(Q,axis=0)


    def sample(self,n_iter):

        np.random.seed(self.seed)
        states, actions, rewards, s_tags, done = [], [], [], [], []

        self.env.reset()
        is_done=False
        for i in range(n_iter):

            state = self.env.state
            state = self.rbf_calc(self.normalization(state))

            if is_done or self.t%500==0:
                self.env.reset()
                state = self.env.state
                state = self.rbf_calc(self.normalization(state))


            if self.exploration.value(self.t)*2>np.random.rand(1)[0]:
                act=self.env.action_space.sample()
                self.t+=1
            else:
                act=self.greedy_policy(state,[False])[0]
                self.t += 1

            s_tag, reward, is_done, _ = self.env.step(act)

            s_tag=self.rbf_calc(self.normalization(s_tag))

            states.append(state)
            actions.append(act)
            rewards.append(reward)
            s_tags.append(s_tag)
            done.append(is_done)


        self.replay_buffer.push(states,actions,rewards,done,s_tags)

    def step(self):

        [states, actions, rewards, done, s_tags]=self.replay_buffer.pop(self.batch_size)

        old=self.w
        a_greedy=self.greedy_policy(s_tags,done)
        phi=self.create_phi(states,actions,[0]*self.batch_size)
        phi_tag=self.create_phi(s_tags,a_greedy,done)
        Q=self.calc_Q(phi)
        Q_tag=self.calc_Q(phi_tag)
        y=(rewards+self.gamma*Q_tag)

        for i in range(self.learning_freq):
            self.w=self.w+self.learning_rate*np.mean(phi.T*(y-Q),axis=1)
            Q = self.calc_Q(phi)

        self.w_save.append(np.sum(self.w))

    def check(self,iter=10,render=True,seed=1):
        rew = []
        done = False
        self.env.seed(seed)
        for i in range(iter):
            self.env.reset()
            rewards = 0
            itr = 0
            state = env.state

            while True:

                if render:
                    env.render()

                state = self.rbf_calc(self.normalization(state))
                # action = int(np.sign(state[1]) + 1)
                # w=np.array([-0.1,0,0,0.1,0.1,0])
                action= self.greedy_policy(state,[False])[0]
                state, reward, done, _ = env.step(action)
                rewards += reward
                if done or itr > 10e2:
                    break
                itr += 1

            rew.append(rewards)

        return np.mean(rew), np.std(rew)

if __name__ == '__main__':
    env = MountainCarWithResetEnv()
    # # run no force
    # env.reset()
    # env.render()
    # is_done = False
    # while not is_done:
    #     _, r, is_done, _ = env.step(1)
    #     env.render()
    #     print(r)
    # # run random forces
    # env.reset()
    # env.render()
    # is_done = False
    # while not is_done:
    #     _, r, is_done, _ = env.step(env.action_space.sample())  # take a random action
    #     env.render()
    #     print(r)
    # set specific
    #batch_size = 1024
    seed_num=5
    #seeds = np.random.randint(1, 100, seed_num)
    seeds=[64,47,93,73,5]
    exp_name = 'MountainCar'
    itertation = 20

    logdir= exp_name + '_' + 'Q-learning' + '_' + "Num_iter"+str(itertation)+"2_epsilon_del"
    logdir= os.path.join('data', logdir)
    os.makedirs(logdir)


    sample=5000
    q_learner_obj=QLearner(env)

    File_object2 =open(osp.join(logdir, "log.txt"), 'w')
    File_object2.writelines("iteration,")
    for j in seeds:
        File_object2.writelines("seeds" + str(j)+",")
    for i in range(itertation):
        if i%10==0:
            print("------- %i------"%i)
            File_object2.writelines("\n" + str(i) + ",")
            for iter in range(seed_num):
                mean,_=q_learner_obj.check(seed=int(seeds[iter]),render=False)
                File_object2.writelines(str(mean) + ",")

            print("mean is {}".format(mean))
        q_learner_obj.sample(sample)
        q_learner_obj.step()

    #q_learner_obj.check()
    File_object2.close()
    print(q_learner_obj.w)
    plt.plot(q_learner_obj.w_save)
    plt.show()
    env.close()








