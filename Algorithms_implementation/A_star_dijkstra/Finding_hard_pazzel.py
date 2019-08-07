from puzzle import *
from planning_utils import *
import heapq
import datetime
from a_star import *
import numpy as np
import dijkstra as dij
import A_star as As

if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    goal_state = initial_state
    length=0
    while length<27:
        path_option=goal_state.get_actions()
        opt=[initial_state.get_manhattan_distance(goal_state.apply_action(act)) for act in path_option]
        if np.random.rand(1)[0]<=0.1:
            goal_state = goal_state.apply_action(np.random.permutation(path_option)[0])
        else:
            goal_state=goal_state.apply_action(path_option[np.argmax(opt)])
        puzzle = Puzzle(initial_state, goal_state)
        length=len(As.solve(puzzle)[0]) - 1

    print(goal_state.to_string())

    solution_start_time = datetime.datetime.now()
    [plane_A_star,states_vis_A]=As.solve(puzzle)

    print('time to solve A_star{}'.format(datetime.datetime.now() - solution_start_time))
    print('length of plan A_star {}'.format(len(plane_A_star)-1))
    print('states visited A_star {}'.format(states_vis_A))
    print("----------------------------")
    solution_start_time = datetime.datetime.now()
    [plane_dijkstra,states_vis_dijkstra]=dij.solve(puzzle)
    print('time to solve dijkstra {}'.format(datetime.datetime.now() - solution_start_time))
    print('length of plan dijkstra {}'.format(len(plane_dijkstra) - 1))
    print('states visited dijkstra {}'.format(states_vis_dijkstra))


