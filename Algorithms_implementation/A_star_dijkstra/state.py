import os


class State:
    def __init__(self, s=None):
        if s is None:
            self._array = [[str(3*i + j) for j in range(3)] for i in range(3)]
        else:
            array = [[c for c in line.split(' ')] for line in s.split(os.linesep)]
            assert len(array) == 3
            for l in array:
                assert len(l) == 3
            self._array = array

    def copy(self):
        result = State()
        result._array = [[c for c in a] for a in self._array]
        return result


    def _get_location_char(self, c):
        for i in range(3):
            for j in range(3):
                if self._array[i][j] == c:
                    return i, j
        assert False

    def _get_empty_location(self):
        return self._get_location_char('0')

    def to_string(self):
        return os.linesep.join([' '.join(l) for l in self._array])

    def __eq__(self, other):
        return self.to_string() == other.to_string()

    def __lt__(self, other):
        return self.to_string() < other.to_string()

    def get_actions(self):
        empty_location = self._get_empty_location()
        actions = []
        if empty_location[0] > 0:
            actions += ['u']
        if empty_location[0] < 2:
            actions += ['d']
        if empty_location[1] > 0:
            actions += ['l']
        if empty_location[1] < 2:
            actions += ['r']
        return actions

    def apply_action(self, a):
        valid_actions = self.get_actions()
        assert a in valid_actions

        new_state = self.copy()

        pos1 = self._get_empty_location()
        pos2 = [pos1[0], pos1[1]]
        if a == 'u':
            pos2[0] -= 1
        elif a == 'd':
            pos2[0] += 1
        elif a == 'l':
            pos2[1] -= 1
        elif a == 'r':
            pos2[1] += 1
        pos2 = tuple(pos2)
        new_state._array[pos2[0]][pos2[1]], new_state._array[pos1[0]][pos1[1]] = new_state._array[pos1[0]][pos1[1]], new_state._array[pos2[0]][pos2[1]]
        return new_state

    def get_manhattan_distance(self, other):
        total_distance = 0
        for i in range(1, 9):
            self_location = self._get_location_char(str(i))
            other_location = other._get_location_char(str(i))
            diff = abs(self_location[0] - other_location[0]) + abs(self_location[1] - other_location[1])
            total_distance += diff
        return total_distance

    def get_dislocation_tile(self, other):
        total_distance = 0
        for i in range(1, 9):
            self_location = self._get_location_char(str(i))
            other_location = other._get_location_char(str(i))
            if self_location!=other_location:
                total_distance += 1
        return total_distance

    def is_same(self, other):
        return self.get_manhattan_distance(other) == 0


if __name__ == '__main__':
    initial_state = State()
    print('initial state')
    print(initial_state.to_string())
    initial_actions = initial_state.get_actions()
    print('actions: {}'.format(initial_actions))
    down_state = initial_state.apply_action('d')
    print('distance to self:')
    print(initial_state.get_manhattan_distance(initial_state))
    print('one down from initial')
    print(down_state.to_string())
    print('distance between both:')
    print(down_state.get_manhattan_distance(initial_state))
    right_state = initial_state.apply_action('r')
    print('one to the right from initial')
    print(right_state.to_string())
    print('another down')
    print((down_state.apply_action('d')).to_string())
    print('distance')
    print((down_state.apply_action('d')).get_manhattan_distance(initial_state))
