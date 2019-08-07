from puzzle import *
from planning_utils import *
import heapq
import datetime


def a_star(puzzle):
    '''
    apply a_star to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.
    alpha_mult=1
    initial = puzzle.start_state
    goal = puzzle.goal_state

    # this is the heuristic function for of the start state
    initial_to_goal_heuristic = initial.get_manhattan_distance(goal)*alpha_mult
    #initial_to_goal_heuristic = initial.get_dislocation_tile(goal) #incorrect tiles heuristic

    # the fringe is the queue to pop items from
    fringe = [(initial_to_goal_heuristic, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}

    while len(fringe) > 0:
        current_priority, current_item = heapq.heappop(fringe)
        if current_item.get_manhattan_distance(goal) == 0:
            return prev

        concluded.add(current_item.to_string())

        actions = current_item.get_actions()
        for act in actions:
            temp=current_item.apply_action(act)
            if temp.to_string() in concluded:
                continue

            temp_score=current_priority-current_item.get_manhattan_distance(goal)*alpha_mult+1
            #temp_score = current_priority - current_item.get_dislocation_tile(goal) + 1 #incorrect tiles heuristic

            if distances.get(temp.to_string()) and distances.get(temp.to_string())<temp_score:
                continue

            if not temp.to_string() in fringe:
                heapq.heappush(fringe,(temp_score+temp.get_manhattan_distance(goal)*alpha_mult,temp))
                #heapq.heappush(fringe, (temp_score + temp.get_dislocation_tile(goal), temp)) #incorrect tiles heuristic

            prev.update({temp.to_string(): current_item})
            distances.update({temp.to_string(): temp_score})
            #distances.update({current_item.to_string(): current_priority})

    return prev


def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = a_star(puzzle)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    #print_plan(plan)
    return [plan,len(prev_mapping)]


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u'
    ]
    #actions = ['r', 'r', 'd', 'l', 'u', 'l']
    goal_state = initial_state

    for a in actions:
        goal_state = goal_state.apply_action(a)
    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    [plan,state_space]=solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))
    print('Plan length {}'.format(len(plan)-1))
    print('Plan state space {}'.format(state_space))
    print_plan(plan)
