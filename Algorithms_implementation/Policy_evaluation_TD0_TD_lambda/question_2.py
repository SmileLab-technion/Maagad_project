import numpy as np
import matplotlib.pyplot as plt
import copy

class states_space(object):

    def __init__(self,
                 expectation=None,
                 cost=None,
                 ):
        self.e_state={}
        self.c_state={}
        self.current_value=None
        self.initial(expectation,cost)
        self.actions_space=np.sum([1 if not i==0 else 0 for i in cost])
        self.Possible_Actions=self.possible_actions(expectation,cost)
        self.policy=policy(cost)
        self.current_value=self.intial_value()
        self.number_of_visits=0
        self.value_simulator=0
        self.q=[0,0,0,0,0]

    def initial(self,expectation,cost):

        for i in range(len(expectation)):
            self.e_state.update({i:expectation[i]})
            self.c_state.update({i: cost[i]})

    def finish_job(self,state_name):

        self.c_state[state_name]=0

    def is_finish(self):
        sum=np.sum([i for i in self.c_state.values()])

        return True if sum == 0 else False

    def return_cost(self,state_name):
        return self.c_state.get(state_name)

    def return_expectation(self,state_name):
        return self.e_state.get(state_name)

    def total_cost(self):
        return np.sum([r for r in self.c_state.values()])

    def return_state(self):
        return self.c_state.keys()

    def possible_actions(self,expectation,cost):

        if np.sum(cost)==0:
            return 0
        else:
            p_act=list(np.zeros((len(cost))))
            for i in range(len(cost)):
                temp_cost = copy.copy(cost)
                if not cost[i]==0:
                    temp_cost[i]=0
                    A=states_space(expectation,temp_cost)
                    p_act[i]=A
            return p_act

    def intial_value(self):

        if not self.current_value is None:
            return self.current_value

        cost=np.sum([r for r in self.c_state.values()])
        r=self.c_state.values()
        if cost==0:
            return 0
        a=self.policy.action
        mu=self.return_expectation(a)
        v_s=cost/mu+self.Possible_Actions[a].intial_value()

        return v_s

    def best_action_current_policy(self):

        possible_state=self.Possible_Actions
        cost=self.total_cost()
        for i in range(len(possible_state)):
            if not possible_state[i]==0:
                mu=self.return_expectation(i)
                v_tag=possible_state[i].current_value
                v_temp=cost/mu+v_tag
                if self.current_value>v_temp:
                    self.current_value=v_temp
                    self.policy.action=i

    def policy_iteration(self):

        if self.total_cost()==0:
            return 0

        possible_state = self.Possible_Actions
        self.best_action_current_policy()
        for state_tag in possible_state:
            if not state_tag==0:
                state_tag.policy_iteration()


    def print_actions(self):
        temp=[]
        if self.total_cost()==0:
            return []
        action=self.policy.action
        temp.append(action)
        return temp+self.Possible_Actions[action].print_actions()

    def simulator(self,action):
        #self.number_of_visits+=1
        state=self
        cost=self.total_cost()
        mu=self.return_expectation(action)
        random=np.random.rand(1)[0]



        if random<mu:
            state=self.Possible_Actions[action]

        return state,cost

    def td_0(self,alpha_d):


        if not self.Possible_Actions==0:

            self.number_of_visits+=1
            visits=self.number_of_visits
            alpha_choice ={0:1/visits,1:0.01,2:10/(100+visits)}
            alpha=alpha_choice[alpha_d]
            action = self.policy.action
            [state_tag, cost]=self.simulator(action)

            value_simulations=self.value_simulator
            value_s_tag=state_tag.value_simulator

            self.value_simulator=value_simulations+alpha*(cost+value_s_tag-value_simulations)


            for i in self.Possible_Actions:
                if not i==0:
                    i.td_0(alpha_d)


    def td_lamda(self,lamda,alpha_d):


        if not self.Possible_Actions[self.policy.action].Possible_Actions==0:

            self.number_of_visits+=1
            visits=self.number_of_visits
            alpha_choice ={0:1/visits,1:0.01,2:10/(100+visits)}
            alpha=alpha_choice[alpha_d]
            action = self.policy.action
            [state_tag, cost_1]=self.simulator(action)
            [state_tag_tage,cost_2]=state_tag.simulator(state_tag.policy.action)

            value_simulations_1=self.value_simulator
            value_simulations_2=state_tag.value_simulator
            value_simulation_3=state_tag_tage.value_simulator

            delta_1=(1-lamda)*(cost_1+value_simulations_2-value_simulations_1)
            delta_2=lamda*(1-lamda)*(cost_1+cost_2+value_simulation_3-value_simulations_1)

            self.value_simulator=value_simulations_1+alpha*(delta_1+delta_2)


            for i in self.Possible_Actions:
                if not i==0:
                    i.td_lamda(lamda,alpha_d)



    def return_all_simulation_value(self,states={}):
        if self.Possible_Actions == 0:
            return {}

        list_A=[str(r) for r in self.c_state.values()]
        values=''.join(list_A)
        if not values in states.keys():
            states.update({values:self.value_simulator})

        else:
            value=np.max((self.value_simulator,states[values]))
            states.update({values:value})

        for i in self.Possible_Actions:
            if not i == 0:
                states.update(i.return_all_simulation_value(states))

        return states


    def return_all_value(self,states={}):
        if self.Possible_Actions == 0:
            return {}

        list_A=[str(r) for r in self.c_state.values()]
        values=''.join(list_A)
        if not values in states.keys():
            states.update({values:self.current_value})

        else:
            value=np.max((self.current_value,states[values]))
            states.update({values:value})

        for i in self.Possible_Actions:
            if not i == 0:
                states.update(i.return_all_value(states))

        return states

    def return_all_q_policy(self,states={}):
        if self.Possible_Actions == 0:
            return {}

        list_A=[str(r) for r in self.c_state.values()]
        values=''.join(list_A)


        min_q=np.min([self.q[key] if not value==0 else 1e5 for key,value in self.c_state.items()])
        if not values in states.keys():
            states.update({values:min_q})

        else:
            value=np.max((min_q,states[values]))
            states.update({values:value})

        for i in self.Possible_Actions:
            if not i == 0:
                states.update(i.return_all_q_policy(states))

        return states

    def greedy(self,epsilon):
        inside=[]
        if self.Possible_Actions==0:
            return 0
        for i in range(len(self.Possible_Actions)):
            if not self.Possible_Actions[i]==0:
                inside.append(i)

        action=np.argmin([self.q[key] if not value==0 else 1e5 for key,value in self.c_state.items()])
        random = np.random.rand(1)[0]

        if random<epsilon:
            action=np.random.choice(inside)

        self.policy.action=action

        return action

    def Q_learning(self,alpha_d,epsilon):


        if not self.Possible_Actions==0:

            self.number_of_visits+=1
            visits=self.number_of_visits
            alpha_choice ={0:1/visits,1:0.01,2:10/(100+visits)}
            alpha=alpha_choice[alpha_d]
            cost=self.total_cost()


            if not self.actions_space==0:
                for i in range(len(self.Possible_Actions)):
                    Q_s=self.q[i]
                    if self.Possible_Actions[i]==0:
                        Q_s_tag=0
                        self.q[i] = Q_s + alpha * (cost + Q_s_tag - Q_s)
                    else:
                        [s_tag,_]=self.simulator(i)

                        if s_tag==0:
                            Q_s_tag=0
                        else:
                            Q_s_tag=s_tag.q[s_tag.greedy(epsilon)]

                        Q_s=self.q[i]
                        self.q[i] = Q_s + alpha * (cost + Q_s_tag - Q_s)
                        self.Possible_Actions[i].Q_learning(alpha_d, epsilon)






class policy(object):

    def __init__(self,cost):
        self.action=self.intial_action(cost)



    def intial_action(self,cost):
        cost=np.argmax(cost)

        return cost








def value_current_policy(state_space,intial_action=None):
    if (state_space.is_finish()):
        return 0

    temp_policy=policy(state_space,policy_user=intial_action)

    value=temp_policy.state_space.total_cost()
    action=temp_policy.action()

    state_space.finish_job(action)
    mu=state_space.return_expectation(action)
    total_value=value/mu+value_current_policy(state_space)

    return total_value


def policy_iteration(state_space):
    value={}
    max_value=1e5
    max_state=0
    for i in state_space.return_state():
        temp_state_space=copy.deepcopy(state_space)
        value_temp=value_current_policy(temp_state_space,i)
        if value_temp<max_value:
            max_value=value_temp
            max_state=copy.deepcopy(state_space)

    return max_value,max_state


def print_policy_iteration(States_Space):
    epsilon = 1e-3
    old_value = States_Space.current_value
    new_value = 1e5
    i = 1
    while np.abs(new_value - old_value) > epsilon:
        print("step {}:".format(i))
        old_value = States_Space.current_value
        print("value is current {}".format(States_Space.current_value))
        print("actions are current {}".format(States_Space.print_actions()))
        States_Space.policy_iteration()
        new_value = States_Space.current_value
        i += 1

    return States_Space


def print_TD_0(expectation,cost,n_iter):

    States_Space_0 = states_space(expectation, cost)
    States_Space_1 = states_space(expectation, cost)
    States_Space_2 = states_space(expectation, cost)

    simulator_value_0 = []
    simulator_value_1 = []
    simulator_value_2 = []

    simulator_value_0_inf = []
    simulator_value_1_inf = []
    simulator_value_2_inf = []

    for i in range(n_iter):
        States_Space_0.td_0(0)
        States_Space_1.td_0(1)
        States_Space_2.td_0(2)
        simulator_value_0.append(np.abs(States_Space_0.value_simulator - States_Space_0.current_value))
        simulator_value_1.append(np.abs(States_Space_1.value_simulator - States_Space_0.current_value))
        simulator_value_2.append(np.abs(States_Space_2.value_simulator - States_Space_0.current_value))

        all_value=np.array([i for i in (States_Space_0.return_all_value({})).values()])
        simulated0 = np.array([j for j in (States_Space_0.return_all_simulation_value({})).values()])
        simulated1 = np.array([n for n in (States_Space_1.return_all_simulation_value({})).values()])
        simulated2 = np.array([r for r in (States_Space_2.return_all_simulation_value({})).values()])

        simulator_value_0_inf.append(np.max(np.abs(all_value - simulated0)))
        simulator_value_1_inf.append(np.max(np.abs(all_value - simulated1)))
        simulator_value_2_inf.append(np.max(np.abs(all_value - simulated2)))


    plt.figure(0, figsize=(8, 8))
    plt.plot(simulator_value_0, label=r'alpha=$\frac{1}{visits}$', markersize=1.2)
    plt.plot(simulator_value_1, label=r'alpha=0.01', markersize=1.2)
    plt.plot(simulator_value_2, label=r'alpha=$\frac{10}{100+visits}$', markersize=1.2)
    # plt.plot(V_TD_inf_norm)
    # plt.title(r'alpha=$\frac{10}{100+number\, of \, visits}$', fontsize=18)
    plt.xlabel("number of steps", fontsize=18)
    plt.ylabel("V_pi(s0)-V_TD(s0)", fontsize=18)
    plt.legend(loc='lower left')
    plt.show()

    plt.figure(1, figsize=(8, 8))
    plt.plot(simulator_value_0_inf, label=r'alpha=$\frac{1}{visits}$', markersize=1.2)
    plt.plot(simulator_value_1_inf, label=r'alpha=0.01', markersize=1.2)
    plt.plot(simulator_value_2_inf, label=r'alpha=$\frac{10}{100+visits}$', markersize=1.2)
    # plt.plot(V_TD_inf_norm)
    # plt.title(r'alpha=$\frac{10}{100+number\, of \, visits}$', fontsize=18)
    plt.xlabel("number of steps", fontsize=18)
    plt.ylabel("||V_pi(s)-V_TD(s)||_inf", fontsize=18)
    plt.legend(loc='lower left')
    plt.show()


def print_TD_lamda(expectation,cost,n_iter,lamda):

    States_Space_0 = states_space(expectation, cost)
    States_Space_1 = states_space(expectation, cost)
    States_Space_2 = states_space(expectation, cost)

    simulator_value_0 = []
    simulator_value_1 = []
    simulator_value_2 = []

    simulator_value_0_inf = []
    simulator_value_1_inf = []
    simulator_value_2_inf = []

    for i in range(n_iter):
        States_Space_0.td_lamda(lamda,0)
        States_Space_1.td_lamda(lamda,1)
        States_Space_2.td_lamda(lamda,2)
        simulator_value_0.append(np.abs(States_Space_0.value_simulator - States_Space_0.current_value))
        simulator_value_1.append(np.abs(States_Space_1.value_simulator - States_Space_0.current_value))
        simulator_value_2.append(np.abs(States_Space_2.value_simulator - States_Space_0.current_value))

        all_value=np.array([i for i in (States_Space_0.return_all_value({})).values()])
        simulated0 = np.array([j for j in (States_Space_0.return_all_simulation_value({})).values()])
        simulated1 = np.array([n for n in (States_Space_1.return_all_simulation_value({})).values()])
        simulated2 = np.array([r for r in (States_Space_2.return_all_simulation_value({})).values()])

        simulator_value_0_inf.append(np.max(np.abs(all_value - simulated0)))
        simulator_value_1_inf.append(np.max(np.abs(all_value - simulated1)))
        simulator_value_2_inf.append(np.max(np.abs(all_value - simulated2)))


    return [simulator_value_0,simulator_value_1,simulator_value_2,simulator_value_0_inf,simulator_value_1_inf,simulator_value_2_inf]


def print_TD_lamda_n_times(expectation,cost,n_iter):

    simulator_value_2_0_all = []
    simulator_value_2_1_all = []
    simulator_value_2_2_all = []

    simulator_value_0_inf_0_all = []
    simulator_value_0_inf_1_all = []
    simulator_value_0_inf_2_all = []

    for i in range(20):
        [simulator_value_0_0, simulator_value_1_0, simulator_value_2_0, simulator_value_0_inf_0,
         simulator_value_1_inf_0,
         simulator_value_2_inf_0] = print_TD_lamda(expectation, cost, n_iter, 0.3)

        [simulator_value_0_1, simulator_value_1_1, simulator_value_2_1, simulator_value_0_inf_1,
         simulator_value_1_inf_1,
         simulator_value_2_inf_1] = print_TD_lamda(expectation, cost, n_iter, 0.5)

        [simulator_value_0_2, simulator_value_1_2, simulator_value_2_2, simulator_value_0_inf_2,
         simulator_value_1_inf_2,
         simulator_value_2_inf_2] = print_TD_lamda(expectation, cost, n_iter, 0.7)

        simulator_value_2_0_all.append(simulator_value_2_0)
        simulator_value_2_1_all.append(simulator_value_2_1)
        simulator_value_2_2_all.append(simulator_value_2_2)

        simulator_value_0_inf_0_all.append(simulator_value_2_inf_0)
        simulator_value_0_inf_1_all.append(simulator_value_2_inf_1)
        simulator_value_0_inf_2_all.append(simulator_value_2_inf_2)

    simulator_value_2_0_all = np.average(np.array(simulator_value_2_0_all), axis=0)
    simulator_value_2_1_all = np.average(np.array(simulator_value_2_1_all), axis=0)
    simulator_value_2_2_all = np.average(np.array(simulator_value_2_2_all), axis=0)

    simulator_value_0_inf_0_all = np.average(np.array(simulator_value_0_inf_0_all), axis=0)
    simulator_value_0_inf_1_all = np.average(np.array(simulator_value_0_inf_1_all), axis=0)
    simulator_value_0_inf_2_all = np.average(np.array(simulator_value_0_inf_2_all), axis=0)

    plt.figure(0, figsize=(8, 8))
    plt.plot(simulator_value_2_0_all, label=r'$\lambda=0.3$', markersize=1.2)
    plt.plot(simulator_value_2_1_all, label=r'$\lambda=0.5$', markersize=1.2)
    plt.plot(simulator_value_2_2_all, label=r'$\lambda=0.7$', markersize=1.2)
    # plt.plot(V_TD_inf_norm)
    # plt.title(r'alpha=$\frac{10}{100+number\, of \, visits}$', fontsize=18)
    plt.xlabel("number of steps", fontsize=18)
    plt.ylabel("V_pi(s0)-V_TD(s0)", fontsize=18)
    plt.legend(loc='lower left')
    plt.show()

    plt.figure(1, figsize=(8, 8))
    plt.plot(simulator_value_0_inf_0_all, label=r'$\lambda=0.3$', markersize=1.2)
    plt.plot(simulator_value_0_inf_1_all, label=r'$\lambda=0.5$', markersize=1.2)
    plt.plot(simulator_value_0_inf_2_all, label=r'$\lambda=0.7$', markersize=1.2)

    # plt.plot(V_TD_inf_norm)
    # plt.title(r'alpha=$\frac{10}{100+number\, of \, visits}$', fontsize=18)
    plt.xlabel("number of steps", fontsize=18)
    plt.ylabel("||V_pi(s)-V_TD(s)||_inf", fontsize=18)
    plt.legend(loc='lower left')
    plt.show()

def q_learning(expectation,cost,n_iter,epsilon):

    States_Space_0 = states_space(expectation, cost)
    States_Space_1 = states_space(expectation, cost)
    States_Space_2 = states_space(expectation, cost)

    States_Space_optimal = print_policy_iteration(States_Space_0)
    v_star=np.array([i for i in States_Space_optimal.return_all_value().values()])

    simulator_value_0 = []
    simulator_value_1 = []
    simulator_value_2 = []

    simulator_value_0_inf = []
    simulator_value_1_inf = []
    simulator_value_2_inf = []

    for i in range(n_iter):
        States_Space_0.Q_learning(0,epsilon)
        States_Space_1.Q_learning(1,epsilon)
        States_Space_2.Q_learning(2,epsilon)

        simulator_value_0.append(np.abs(np.min(States_Space_0.q) -v_star[0]))
        simulator_value_1.append(np.abs(np.min(States_Space_1.q) -v_star[0]))
        simulator_value_2.append(np.abs(np.min(States_Space_2.q) -v_star[0]))

        simulated0 = np.array([j for j in (States_Space_0.return_all_q_policy({})).values()])
        simulated1 = np.array([n for n in (States_Space_1.return_all_q_policy({})).values()])
        simulated2 = np.array([r for r in (States_Space_2.return_all_q_policy({})).values()])

        simulator_value_0_inf.append(np.max(np.abs(v_star - simulated0)))
        simulator_value_1_inf.append(np.max(np.abs(v_star - simulated1)))
        simulator_value_2_inf.append(np.max(np.abs(v_star - simulated2)))


    return [simulator_value_0,simulator_value_1,simulator_value_2,simulator_value_0_inf,simulator_value_1_inf,simulator_value_2_inf]

def print_q_learning(expectation,cost,n_iter,epsilon):
    [simulator_value_0, simulator_value_1, simulator_value_2, simulator_value_0_inf, simulator_value_1_inf,
     simulator_value_2_inf]=q_learning(expectation,cost,n_iter,epsilon)


    plt.figure(0, figsize=(8, 8))
    plt.plot(simulator_value_0, label=r'alpha=$\frac{1}{visits}$', markersize=1.2)
    plt.plot(simulator_value_1, label=r'alpha=0.01', markersize=1.2)
    plt.plot(simulator_value_2, label=r'alpha=$\frac{10}{100+visits}$', markersize=1.2)
    # plt.plot(V_TD_inf_norm)
    # plt.title(r'alpha=$\frac{10}{100+number\, of \, visits}$', fontsize=18)
    plt.xlabel("number of steps", fontsize=18)
    plt.ylabel("V_pi(s0)-V_TD(s0)", fontsize=18)
    plt.legend(loc='lower left')
    plt.show()

    plt.figure(1, figsize=(8, 8))
    plt.plot(simulator_value_0_inf, label=r'alpha=$\frac{1}{visits}$', markersize=1.2)
    plt.plot(simulator_value_1_inf, label=r'alpha=0.01', markersize=1.2)
    plt.plot(simulator_value_2_inf, label=r'alpha=$\frac{10}{100+visits}$', markersize=1.2)
    # plt.plot(V_TD_inf_norm)
    # plt.title(r'alpha=$\frac{10}{100+number\, of \, visits}$', fontsize=18)
    plt.xlabel("number of steps", fontsize=18)
    plt.ylabel("||V_pi(s)-V_TD(s)||_inf", fontsize=18)
    plt.legend(loc='lower left')
    plt.show()


def print_q_learning_epsilon_compare(expectation,cost,n_iter,epsilon1,epsilon2):
    [_, _, simulator_value_2_epsilon_1, _, _,
     simulator_value_2_inf_epsilon_1]=q_learning(expectation,cost,n_iter,epsilon1)

    [_, _, simulator_value_2_epsilon_2, _, _,
     simulator_value_2_inf_epsilon_2]=q_learning(expectation,cost,n_iter,epsilon2)


    plt.figure(0, figsize=(8, 8))
    plt.plot(simulator_value_2_epsilon_1, label=r'$\epsilon=$'+str(epsilon1), markersize=1.2)
    plt.plot(simulator_value_2_epsilon_2, label=r'$\epsilon=$'+str(epsilon2), markersize=1.2)
    # plt.plot(V_TD_inf_norm)
    # plt.title(r'alpha=$\frac{10}{100+number\, of \, visits}$', fontsize=18)
    plt.xlabel("number of steps", fontsize=18)
    plt.ylabel("V_pi(s0)-V_TD(s0)", fontsize=18)
    plt.legend(loc='lower left')
    plt.show()

    plt.figure(1, figsize=(8, 8))
    plt.plot(simulator_value_2_inf_epsilon_1,  label=r'$\epsilon=$'+str(epsilon1), markersize=1.2)
    plt.plot(simulator_value_2_inf_epsilon_2,  label=r'$\epsilon=$'+str(epsilon2), markersize=1.2)
    # plt.plot(V_TD_inf_norm)
    # plt.title(r'alpha=$\frac{10}{100+number\, of \, visits}$', fontsize=18)
    plt.xlabel("number of steps", fontsize=18)
    plt.ylabel("||V_pi(s)-V_TD(s)||_inf", fontsize=18)
    plt.legend(loc='lower left')
    plt.show()

def main():
    n_iter=1000
    expectation=[0.6,0.5,0.3,0.7,0.1]
    cost = [1, 4, 6, 2, 9]

    States_Space = states_space(expectation, cost)

    File_object2 = open("V_polocy.txt", "w")
    for state,value in States_Space.return_all_value({}).items():
        File_object2.writelines(str(state)+","+str(value)+"\n")
    File_object2.close()
#----section d policy iteration  -----------
    #print_policy_iteration(States_Space)

#------section g policy evaluation using TD(0)  -----------
    #print_TD_0(expectation,cost,n_iter)

#-----------section h policy evaluation using TD(lamda)  -----------
    #print_TD_lamda_n_times(expectation,cost,n_iter)

#-------------section i policy evaluation using TD(lamda)----------------
    # print_q_learning(expectation, cost, n_iter, 0.1)
    #print_q_learning_epsilon_compare(expectation, cost, n_iter, 0.1, 0.01)





if __name__ == "__main__":
    main()








