import itertools

from gym import core, spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding

import numpy as np


class ArmExtensionEnv(core.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    SAFE_ZONE = 0.25
    BALL_RADIUS = 0.1
    UPPER_ARM_LENGTH = 1.0
    LOWER_ARM_LENGTH = 1.0
    ANGULAR_ACCELERATION = 0.01
    ANGLE_CONTROL = [-ANGULAR_ACCELERATION, 0, ANGULAR_ACCELERATION]
    ACTIONS = list(itertools.product(ANGLE_CONTROL, ANGLE_CONTROL))

    def __init__(self):
        self.viewer = None
        high = np.array([
            np.pi, np.pi,
            self.ANGULAR_ACCELERATION, self.ANGULAR_ACCELERATION
        ])
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Discrete(9)
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.traces = []
        self.state = np.array([
            np.pi * 1 / 8.,  # Angular position of joint 1
            np.pi * 6 / 8.,  # Angular position of joint 2
            0, 0  # Angular velocities of the joints
        ])
        return self.state

    def _step(self, a):
        # Request an action from the agent
        v1, v2 = self.ACTIONS[a]

        # Step the simulation
        self.state = self.state.copy()
        self.state[2] += v1
        self.state[3] += v2
        self.state[0] += self.state[2]
        self.state[1] += self.state[3]

        # Calculate the reward
        x, y = self._get_position()
        distance_to_ball_sq = np.sqrt((2 - x) ** 2 + y ** 2) - self.BALL_RADIUS
        self.traces.append((x, y))
        if y > self.SAFE_ZONE or y < -self.SAFE_ZONE:
            reward = -100
        elif distance_to_ball_sq <= 0:
            reward = 100
        else:
            # Find the distance to the surface of the ball
            reward = -distance_to_ball_sq
        return (
            np.array(self.state),
            reward,
            abs(reward) == 100,  # Stop early if we win or lose
            {}
        )

    def _get_position(self):
        """Return the (x, y) position of the arm's end."""
        a0 = self.state[0]
        a1 = self.state[1]
        x = np.sin(a0) + np.sin(a1 + a0)
        y = -np.cos(a0) - np.cos(a1 + a0)
        return x, y

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-.2, 2.2, -1.2, 1.2)

        # Do some calculations
        # This code is derived from the Acrobot sample
        a0 = self.state[0]
        a1 = self.state[1]
        p1 = [-self.UPPER_ARM_LENGTH *
              np.cos(a0), self.UPPER_ARM_LENGTH * np.sin(a0)]
        p2 = [p1[0] - self.LOWER_ARM_LENGTH * np.cos(a0 + a1),
              p1[1] + self.LOWER_ARM_LENGTH * np.sin(a0 + a1)]
        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [a0 - np.pi / 2, a0 + a1 - np.pi / 2]

        # Draw the path taken
        if len(self.traces) >= 2:
            for i in range(len(self.traces) - 1):
                x1, y1 = self.traces[i]
                x2, y2 = self.traces[i + 1]
                line = self.viewer.draw_line((x1, y1), (x2, y2))
                line.set_color(0, .8, .8)

        # Draw danger lines
        l1 = self.viewer.draw_line((-2.2, -self.SAFE_ZONE), (2.2, -self.SAFE_ZONE))
        l2 = self.viewer.draw_line((-2.2, self.SAFE_ZONE), (2.2, self.SAFE_ZONE))
        l1.set_color(.8, 0, 0)
        l2.set_color(.8, 0, 0)

        # Draw the target circle
        transform = rendering.Transform(rotation=0, translation=(2.0, 0.0))
        circle = self.viewer.draw_circle(self.BALL_RADIUS)
        circle.set_color(.8, .8, 0)
        circle.add_attr(transform)

        # Draw the mechanism
        # This code is also derived from the Acrobot sample
        for ((x, y), th) in zip(xys, thetas):
            l, r, t, b = 0, 1, .1, -.1
            transform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(transform)
            link.set_color(.8, .8, .8)
            circle = self.viewer.draw_circle(.1)
            circle.set_color(.4, .4, .4)
            circle.add_attr(transform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
