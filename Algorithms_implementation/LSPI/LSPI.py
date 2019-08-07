import numpy as np
from mountain_car_with_data_collection import MountainCarWithResetEnv
import matplotlib.pyplot as plt
import os
import time
import os.path as osp

def sample(env):

    [min_position,max_position]=env.min_position,env.max_position

    [min_velocity,max_velocity]=-env.max_speed,env.max_speed

    [position_seed,velocity_seed]=(np.random.uniform(min_position,max_position),(np.random.uniform(min_velocity,max_velocity)))

    state=np.array((position_seed,velocity_seed))
    action=np.random.randint(env.action_space.n)
    env.reset_specific(state[0],state[1])
    s_tag, reward,is_done, _ = env.step(action)

    return [state,action,reward,s_tag,is_done]

def create_database(env,n_iter=100000):
    states, actions, rewards, s_tags,done = [], [], [], [],[]

    for i in range(n_iter):
        [state, action, reward, s_tag,is_done] = sample(env)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        s_tags.append(s_tag)
        done.append(is_done)

    mean=np.mean(states,axis=0)
    std=np.std(states,axis=0)

    states=(states - mean) / std
    s_tag=(s_tags - mean) / std

    database={"states": np.array(states),
              "actions": np.array(actions),
              "reward": np.array(rewards),
              "s_tag":  np.array(s_tag),
              "done": np.array(done)}

    return database,mean,std


def grid(grid_num,max):

    num=2*max/(grid_num-1)
    x = np.arange(-max-1e-8,max+1e-8, num)
    y = np.arange(-max-1e-8, max+1e-8, num)
    sigma=x[1]-x[0]
    xx, yy = np.meshgrid(x, y, sparse=False)
    xx = np.reshape(xx, (xx.shape[0]*xx.shape[1], 1))
    yy = np.reshape(yy, (yy.shape[0]*yy.shape[1], 1))
    grider=np.concatenate((xx,yy),axis=1)

    return grider,sigma

def get_grid(env, nFeatures=4):
    assert np.sqrt(nFeatures) == int(np.sqrt(nFeatures))

    grid_size = int(np.log2(nFeatures))

    s = env.min_position + (env.max_position - env.min_position) / (grid_size + 1)
    pos_grid = np.linspace(s, env.max_position, grid_size, endpoint=False)

    s = -env.max_speed + (env.max_speed - -env.max_speed) / (grid_size + 1)
    vel_grid = np.linspace(s, env.max_speed, grid_size, endpoint=False)
    centers = np.zeros((nFeatures, 2))
    idx = 0

    for i in range(grid_size):
        for j in range(grid_size):
            centers[idx, :] = [pos_grid[i], vel_grid[j]]
            idx += 1

    return centers,1

def rbf_proj(env,database,grid_num=2):
    states = np.array(database["states"])
    s_tag = np.array(database["s_tag"])

    grider,rbf_sigma=grid(grid_num,np.max(states))

    states=rbf_calc(states,grider,rbf_sigma)
    s_tag=rbf_calc(s_tag,grider,rbf_sigma)

    database.update({"states":np.array(states)})
    database.update({"s_tag": np.array(s_tag)})

    return database,grider,rbf_sigma

def rbf_calc(x,grider,rbf_sigma):
    all_states=[]
    rbf_sigma=2 * rbf_sigma ** 2
    for mean in grider:
        new_state=np.exp((-np.linalg.norm(x -mean,2,axis=-1) ** 2) / rbf_sigma)
        all_states.append(new_state)
    new_state=(new_state-new_state+1)
    all_states.append(new_state)
    return np.array(all_states).T

def create_phi(state,action,done,action_space=3):
    state_space=len(state)
    phi=np.zeros((state_space*action_space,1))
    phi[action*state_space:(action+1)*state_space,0]=state*(1-done)
    return phi

def greedy_policy(state,w,done):
    phi_state=np.concatenate((create_phi(state,0,done),create_phi(state,1,done)),axis=1)
    phi_state=np.concatenate((phi_state,create_phi(state,2,done)),axis=1)
    Q=np.dot(phi_state.T,w)



    return np.argmax(Q),Q


def calc_c(batch):
    B=[]
    for i in range(batch["states"].shape[0]):
        [state, action, reward, _,done] = batch["states"][i, :], batch["actions"][i], batch["reward"][i], batch["s_tag"][i, :],batch["done"][i]
        phi = create_phi(state, action,0)
        B.append(reward * phi)

    return np.mean(B,axis=0)


def calc_d(batch,w,gamma=0.999):
    A=[]
    phis=[]
    a_all=[]
    Q_0_all=[]
    s_tag_s=[]
    for i in range(batch["states"].shape[0]):
        [state, action, _, s_tag,done] = batch["states"][i, :], batch["actions"][i], batch["reward"][i], batch["s_tag"][i, :],batch["done"][i]
        phi=create_phi(state,action,0)
        a_greedy,Q_0=greedy_policy(s_tag,w,done)
        Q_0_all.append(Q_0)
        a_all.append(a_greedy)
        phi_tag= create_phi(s_tag, a_greedy,done)
        A.append(np.outer(phi.T, phi.T - gamma * phi_tag.T))
       #A.append(np.dot(phi, (phi.T-gamma * phi_tag.T)))
        phis.append(phi.T)


    return np.mean(A,axis=0)

def check_w(env,w,mean,std,grider,rbf_sigma,seed,iter=10,render=False):


    rew=[]
    done=False
    env.seed(seed)
    for i in range(iter):
        env.reset()
        rewards=0
        itr=0
        state = env.state
        while True:

            if render:
                env.render()

            state=(state-mean)/std

            state=rbf_calc(state,grider,rbf_sigma)


            #action = int(np.sign(state[1]) + 1)
            #w=np.array([-0.1,0,0,0.1,0.1,0])
            action, _=greedy_policy(state,w,done)
            state, reward, done, _ = env.step(action)
            rewards+=reward
            if done or itr>10e2:
                break
            itr+=1

        rew.append(rewards)

    return np.mean(rew)


if __name__ == '__main__':
    env = MountainCarWithResetEnv()
    exp_name='MountainCar'
    np.random.seed(1)
    data_collection = 10000
    seed_num=5
    iter=30

    logdir= exp_name + '_' + 'LSPI' + '_' + "Num_iter"+str(data_collection/1000)+"K"+"20iter"
    logdir= os.path.join('data', logdir)
    os.makedirs(logdir)
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

    print("collecting data")
    database,mean_s,std_s = create_database(env, data_collection)
    database,grider,rbf_sigma=rbf_proj(env,database,5)

    #-----------Section B-percentage of possitive reward------
    print((np.sum(database["reward"])/data_collection)*100)

    action_space = env.action_space.n
    state_space = np.array(database["states"]).shape[1]

    w= np.random.randn(action_space * state_space)
    d_k=calc_c(database)
    w_all=[]
    #seeds = np.random.randint(1, 10000, seed_num)
    seeds = [64, 47, 93, 73, 5]
    means=[]
    File_object2 =open(osp.join(logdir, "log.txt"), 'w')
    File_object2.writelines("iteration,")
    for j in seeds:
        File_object2.writelines("seeds" + str(j)+",")
    for i in range(iter):
        print("**********iter %i**********" % i )
        File_object2.writelines("\n"+str(i) + ",")

        for iter in range(seed_num):
            mean=(check_w(env,w,mean_s,std_s,grider,rbf_sigma,int(seeds[iter])))
            File_object2.writelines(str(mean) + ",")

        C_k=calc_d(database,w,gamma= 0.999)
        temp = np.linalg.solve(C_k, d_k)
        # print(w)
        w_all.append(np.sum(temp - w))
        w = temp





    plt.plot(w_all[1:])
    plt.show()




    env.close()