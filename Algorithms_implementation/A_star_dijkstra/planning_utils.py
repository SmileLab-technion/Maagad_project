def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorhmit
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    temp = prev[goal_state.to_string()]
    while temp:
        actions= temp.get_actions()
        for act in actions:
            state_t=temp.apply_action(act)
            if state_t.is_same(goal_state):
                result.append((temp,act))
                goal_state=temp
                temp=prev[temp.to_string()]
                break
    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
