import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class CartPoleContEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, initial_theta=0.0):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.tau = 0.01  # seconds between state updates
        self.initial_theta = initial_theta
        self.planning_steps = 600

        # Angle at which to fail the episode
        self.theta_threshold_radians = np.pi / 8.0
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        high = h=np.array([4.0])
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.planning_steps_counter = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_accelerations(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = action
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        temp1 = force - self.masspole * self.length * theta_dot * theta_dot * sin_theta
        temp2 = self.masscart + self.masspole * sin_theta * sin_theta

        theta_acc = (temp1 * cos_theta + self.gravity * sin_theta * (self.masspole + self.masscart)) / (self.length * temp2)
        x_acc = (temp1 + self.gravity * self.masspole * sin_theta * cos_theta) / temp2

        return theta_acc, x_acc

    def get_state_change(self, state, action):
        theta_acc, x_acc = self._compute_accelerations(state, action)

        state_change = np.array([state[1], x_acc, state[3], theta_acc])
        state_change *= self.tau

        return np.array(self.state) + state_change

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.state = tuple((self.get_state_change(self.state, action[0])).tolist())

        self.planning_steps_counter += 1
        done = self.planning_steps_counter >= self.planning_steps

        reward = 1.0 if np.abs(self.state[2]) < self.theta_threshold_radians else -1.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.planning_steps_counter = 0
        self.state = np.array([0.0, 0.0, self.initial_theta, 0.0])
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        cartx %= screen_width
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = CartPoleContEnv()
    # run no force
    env.reset()

    env.render()
    is_done = False
    while not is_done:
        _, r, is_done, _ = env.step(np.array([0.0]))
        env.render()
        print(r)
    # run random forces
    env.reset()
    env.render()
    is_done = False
    while not is_done:
        _, r, is_done, _ = env.step(env.action_space.sample())  # take a random action
        env.render()
        print(r)
    env.close()
