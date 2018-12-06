"""
Referense:  https://morvanzhou.github.io/tutorials/


Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""
import numpy as np
import pyglet


pyglet.clock.set_fps_limit(10000)


class CarEnv(object):
    n_sensor = 9
    action_dim = 1
    state_dim = n_sensor
    viewer = None
    viewer_xy = (1100, 100)
    sensor_max = 150.
    start_point = [50, 25 + 50 * np.random.randint(2, size=1)]
    speed = 50.
    dt = 0.1

    def __init__(self, discrete_action=False):
        self.is_discrete_action = discrete_action
        if discrete_action:
            self.actions = [-1, 0, 1]
        else:
            self.action_bound = [-1, 1]

        self.terminal = False
        # node1 (x, y, r, w, l),
        self.car_info = np.array([0, 0, 0, 20, 40], dtype=np.float64)   # car coordination

        ob_choice = 0

        self.obstacle_coords = np.array([[
            [200, 10 + 50 * ob_choice],
            [300, 10 + 50 * ob_choice],
            [300, 40 + 50 * ob_choice],
            [200, 40 + 50 * ob_choice],
        ],[
            [400, 60 - 50 * ob_choice],
            [550, 60 - 50 * ob_choice],
            [550, 90 - 50 * ob_choice],
            [400, 90 - 50 * ob_choice],
        ],[
            [650, 10 + 50 * ob_choice],
            [750, 10 + 50 * ob_choice],
            [750, 40 + 50 * ob_choice],
            [650, 40 + 50 * ob_choice],
        ],[
            [850, 60 - 50 * ob_choice],
            [950, 60 - 50 * ob_choice],
            [950, 90 - 50 * ob_choice],
            [850, 90 - 50 * ob_choice],
        ]]

        , dtype=np.float64)

        self.sensor_info = self.sensor_max + np.zeros((self.n_sensor, 3))  # n sensors, (distance, end_x, end_y)

    def step(self, action):
        if self.is_discrete_action:
            action = self.actions[action]
        else:
            action = np.clip(action, *self.action_bound)[0]
        self.car_info[2] += action * np.pi/30  # max r = 6 degree
        self.car_info[:2] = self.car_info[:2] + \
                            self.speed * self.dt * np.array([np.cos(self.car_info[2]), np.sin(self.car_info[2])])

        self._update_sensor()
        s = self._get_state()

        if self.terminal and self.car_info[0] <= 950:
            r = -1
        elif self.terminal and self.car_info[0] > 950:
            r = 1
        else:
            r = 0

        return s, r, self.terminal

    def reset(self):
        self.terminal = False
        self.car_info[:3] = np.array([50, 25 + 50 * np.random.randint(2, size=1), 0])

        self._update_sensor()
        return self._get_state()

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.car_info, self.sensor_info, self.obstacle_coords)
        self.viewer.render()

    def sample_action(self):
        if self.is_discrete_action:
            a = np.random.choice(list(range(3)))
        else:
            a = np.random.uniform(*self.action_bound, size=self.action_dim)
        return a

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self):
        s = self.sensor_info[:, 0].flatten()/self.sensor_max
        return s

    def _update_sensor(self):
        cx, cy, rotation = self.car_info[:3]

        n_sensors = len(self.sensor_info)
        sensor_theta = np.linspace(-np.pi / 4, np.pi / 4, n_sensors)
        xs = cx + (np.zeros((n_sensors, ))+self.sensor_max) * np.cos(sensor_theta)
        ys = cy + (np.zeros((n_sensors, ))+self.sensor_max) * np.sin(sensor_theta)
        xys = np.array([[x, y] for x, y in zip(xs, ys)])    # shape (5 sensors, 2)

        # sensors
        tmp_x = xys[:, 0] - cx
        tmp_y = xys[:, 1] - cy
        # apply rotation
        rotated_x = tmp_x * np.cos(rotation) - tmp_y * np.sin(rotation)
        rotated_y = tmp_x * np.sin(rotation) + tmp_y * np.cos(rotation)
        # rotated x y
        self.sensor_info[:, -2:] = np.vstack([rotated_x+cx, rotated_y+cy]).T

        q = np.array([cx, cy])
        for si in range(len(self.sensor_info)):
            s = self.sensor_info[si, -2:] - q
            possible_sensor_distance = [self.sensor_max]
            possible_intersections = [self.sensor_info[si, -2:]]

            # obstacle collision
            for oj in range(self.obstacle_coords.shape[0]):

                for oi in range(self.obstacle_coords.shape[1]):
                    p = self.obstacle_coords[oj][oi]
                    r = self.obstacle_coords[oj][(oi + 1) % self.obstacle_coords.shape[1]] - self.obstacle_coords[oj][oi]
                    if np.cross(r, s) != 0:  # may collision
                        t = np.cross((q - p), s) / np.cross(r, s)
                        u = np.cross((q - p), r) / np.cross(r, s)
                        if 0 <= t <= 1 and 0 <= u <= 1:
                            intersection = q + u * s
                            possible_intersections.append(intersection)
                            possible_sensor_distance.append(np.linalg.norm(u * s))

            # window collision
            win_coord = np.array([
                [0, 0],
                [self.viewer_xy[0], 0],
                [*self.viewer_xy],
                [0, self.viewer_xy[1]],
                [0, 0],
            ])
            for oi in range(4):
                p = win_coord[oi]
                r = win_coord[(oi + 1) % len(win_coord)] - win_coord[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = p + t * r
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(intersection - q))

            distance = np.min(possible_sensor_distance)
            distance_index = np.argmin(possible_sensor_distance)
            self.sensor_info[si, 0] = distance
            self.sensor_info[si, -2:] = possible_intersections[distance_index]
            if distance < self.car_info[-1]/2 or self.car_info[0] > 950:
                self.terminal = True


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, car_info, sensor_info, obstacle_coords):
        super(Viewer, self).__init__(width, height, resizable=False, caption='2D car', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=150, y=100)
        pyglet.gl.glClearColor(*self.color['background'])

        self.obstacle_coords = obstacle_coords

        self.car_info = car_info
        self.sensor_info = sensor_info

        self.batch = pyglet.graphics.Batch()
        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)

        self.sensors = []
        line_coord = [0, 0] * 2
        c = (73, 73, 73) * 2
        for i in range(len(self.sensor_info)):
            self.sensors.append(self.batch.add(2, pyglet.gl.GL_LINES, foreground, ('v2f', line_coord), ('c3B', c)))

        car_box = [0, 0] * 4

        c = (249, 86, 86) * 4
        self.car = self.batch.add(4, pyglet.gl.GL_QUADS, foreground, ('v2f', car_box), ('c3B', c))

        c = (134, 181, 244) * 4
        for oj in range(self.obstacle_coords.shape[0]):
            self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', obstacle_coords[oj].flatten()), ('c3B', c))

    def render(self):
        pyglet.clock.tick()
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

        # self.fps_display.draw()

    def _update(self):
        cx, cy, r, w, l = self.car_info

        # sensors
        for i, sensor in enumerate(self.sensors):
            sensor.vertices = [cx, cy, *self.sensor_info[i, -2:]]

        # car
        xys = [
            [cx + l / 2, cy + w / 2],
            [cx - l / 2, cy + w / 2],
            [cx - l / 2, cy - w / 2],
            [cx + l / 2, cy - w / 2],
        ]
        r_xys = []
        for x, y in xys:
            tempX = x - cx
            tempY = y - cy
            # apply rotation
            rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
            rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
            # rotated x y
            x = rotatedX + cx
            y = rotatedY + cy
            r_xys += [x, y]
        self.car.vertices = r_xys


if __name__ == '__main__':
    np.random.seed(1)
    env = CarEnv()
    env.set_fps(30)
    for ep in range(20):
        s = env.reset()
        # for t in range(100):
        while True:
            env.render()
            s, r, done = env.step(env.sample_action())
            #s, r, done = env.step([0])
            #print(s)
            if done:
                break