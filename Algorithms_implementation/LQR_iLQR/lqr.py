import numpy as np
from cartpole_cont import CartPoleContEnv
import time
import matplotlib.pyplot as plt, matplotlib.image as mpimg

def get_A(cart_pole_env):
    '''
    create and returns the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau
    A = np.matrix([[0, 1, 0, 0],
                   [0, 0, pole_mass * g / cart_mass, 0],
                   [0, 0, 0, 1],
                   [0, 0, (g / pole_length) * (1 + pole_mass / cart_mass), 0]])

    return (np.identity(4)+A*dt)


def get_B(cart_pole_env):
    '''
    create and returns the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    B = np.matrix([[0],
                   [1/cart_mass],
                   [0],
                   [1/(cart_mass*pole_length)]])

    return B*dt


def find_lqr_control_input(cart_pole_env):
    '''
    implements the LQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action of shape (1,4).
    '''
    assert isinstance(cart_pole_env, CartPoleContEnv)

    # TODO - you first need to compute A and B for LQR
    A = get_A(cart_pole_env)
    B = get_B(cart_pole_env)

    # TODO - Q and R should not be zero, find values that work, hint: all the values can be <= 1.0
    w1=1e-6
    w2=1
    w3=1e-6

    Q = np.matrix([
        [w1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, w2, 0],
        [0, 0, 0, 0]
    ])

    R = np.matrix([w3])

    # TODO - you need to compute these matrices in your solution, but these are not returned.
    Ps = []
    us = []
    xs = [np.expand_dims(cart_pole_env.state, 1)]
    Ks = []

    Ps.append(Q)
    for i in range(cart_pole_env.planning_steps,0,-1):
        P_t_1=Ps.pop()

        temp_1=np.linalg.inv(R+np.dot(np.dot(B.T,P_t_1),B))
        temp_2=np.dot(B.T,np.dot(P_t_1,A))
        temp_3=np.dot(A.T,np.dot(P_t_1,B))
        temp_4 = np.dot(A.T, np.dot(P_t_1, A))

        P_t=Q+temp_4-np.dot(temp_3,np.dot(temp_1,temp_2))
        Ps.append(P_t)
        Ks.append(-np.dot(temp_1,temp_2))

    X_temp=cart_pole_env.state,
    for i in Ks:
        us.append(np.dot(i,xs[-1]))
        x_t_1=cart_pole_env.get_state_change(xs[-1][:,0],us[-1][0,0])
        xs.append(np.expand_dims(x_t_1, 1))


    # TODO - these should be returned see documentation above


    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))

def Finding_unstable():
    # the following is an example to start at a different theta
    # env = CartPoleContEnv(initial_theta=np.pi * 0.25)

    # print the matrices used in LQR
    #print('A: {}'.format(get_A(env)))
    #print('B: {}'.format(get_B(env)))

    valid_episode=True
    for i in range(1,1000,1):
        env = CartPoleContEnv(initial_theta=np.pi * 0.001*i)
    # start a new episode
        actual_state = env.reset()
        #env.render()
        # use LQR to plan controls
        xs, us, Ks = find_lqr_control_input(env)
        # run the episode until termination, and print the difference between planned and actual
        is_done = False
        iteration = 0
        is_stable_all = []
        while not is_done:
            # print the differences between planning and execution time
            predicted_theta = xs[iteration].item(2)
            actual_theta = actual_state[2]
            predicted_action = us[iteration].item(0)
            actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
            #actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
            #print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
            # apply action according to actual state visited
            # make action in range
            actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
            actual_action = np.array([actual_action])
            actual_state, reward, is_done, _ = env.step(actual_action)
            is_stable = reward == 1.0
            is_stable_all.append(is_stable)
            #env.render()
            #time.sleep(0.2)
            iteration += 1
        env.close()
        # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
        valid_episode = np.all(is_stable_all[-100:])
        if not valid_episode:
            break

    print("the unstable angle is {}".format(np.round(180 *i* 0.001,2)))

def ploting_teta(a,b,c,d):
    tetha={str(a):[],str(b):[],str(c):[],str(d):[]}
    for i in [a,b,c,d]:
        env = CartPoleContEnv(initial_theta=np.pi * i)
        # the following is an example to start at a different theta
        # env = CartPoleContEnv(initial_theta=np.pi * 0.25)
        tau=env.tau
        # print the matrices used in LQR
        print('A: {}'.format(get_A(env)))
        print('B: {}'.format(get_B(env)))

        # start a new episode
        actual_state = env.reset()
        # env.render()
        # use LQR to plan controls
        xs, us, Ks = find_lqr_control_input(env)
        # run the episode until termination, and print the difference between planned and actual
        is_done = False
        iteration = 0
        is_stable_all = []
        angular = []
        while not is_done:
            # print the differences between planning and execution time
            predicted_theta = xs[iteration].item(2)
            actual_theta = actual_state[2]
            tetha[str(i)].append(actual_theta)
            predicted_action = us[iteration].item(0)
            actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
            #actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
            #print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
            # apply action according to actual state visited
            # make action in range
            actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
            actual_action = np.array([actual_action])
            actual_state, reward, is_done, _ = env.step(actual_action)
            is_stable = reward == 1.0
            is_stable_all.append(is_stable)
            # env.render()
            # time.sleep(0.2)
            iteration += 1
        env.close()

        # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
        valid_episode = np.all(is_stable_all[-100:])
        # print if LQR succeeded
    t=np.linspace(1, 600, num=600)*tau
    plt.figure(0, figsize=(8, 8))
    plt.plot(t,tetha[str(a)], 'k', markersize=1.2, label='theta_unstable='+str(np.round(a,4))+'pi')
    plt.plot(t,tetha[str(d)], 'g', markersize=1.2, label=str(np.round(d,4))+'pi')
    plt.plot(t,tetha[str(b)], 'r', markersize=1.2, label='0.1pi')
    plt.plot(t,tetha[str(c)], 'b', markersize=1.2, label='0.5pi')

    plt.xlabel("t[sec]", fontsize=18)
    plt.ylabel("deg[rad]", fontsize=18)

    #plt.xlim(0,300)
    plt.legend(loc='upper left')
    plt.show()




if __name__ == '__main__':
    Finding_unstable()
    ploting_teta(19.51/180, 0.1, 0.5,19.50/180)
    env = CartPoleContEnv(initial_theta=np.pi * 0.1)
    # the following is an example to start at a different theta
    # env = CartPoleContEnv(initial_theta=np.pi * 0.25)

    # print the matrices used in LQR
    print('A: {}'.format(get_A(env)))
    print('B: {}'.format(get_B(env)))

    # start a new episode
    actual_state = env.reset()
    #env.render()
    # use LQR to plan controls
    xs, us, Ks = find_lqr_control_input(env)
    # run the episode until termination, and print the difference between planned and actual
    is_done = False
    iteration = 0
    is_stable_all = []
    angular=[]
    while not is_done:
        # print the differences between planning and execution time
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
        #print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
        # apply action according to actual state visited
        # make action in range
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        #env.render()
        #time.sleep(0.2)
        iteration += 1
    env.close()
    # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
    valid_episode = np.all(is_stable_all[-100:])
    # print if LQR succeeded
    print('valid episode: {}'.format(valid_episode))

