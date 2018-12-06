import tensorflow as tf
import numpy as np
import os
import shutil
import random
from arm_env import ArmEnv

np.random.seed(1234)
tf.set_random_seed(1234)

MAX_EPISODES = 500
MAX_EP_STEPS = 200
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 500
REPLACE_ITER_C = 500
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
TAU = 0.001  # soft replacement
VAR_MIN = 0.01
RENDER = False
LOAD = False
MODE = ['easy', 'hard']
SPARSE = True
n_model = 1
use_her = True
K = 4

env = ArmEnv(mode=MODE[n_model], sparse=SPARSE)
STATE_DIM = env.state_dim

ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound


class Episode_experience():
    def __init__(self):
        self.memory = []

    def add(self, state, action, reward, next_state, done, goal):
        self.memory += [(state, action, reward, next_state, done, goal)]

    def clear(self):
        self.memory = []


class DDPG(object):
    def __init__(self, sess, a_dim, s_dim, a_bound, ):
        self.memory = []
        self.memory_her = []
        self.pointer = 0
        self.sess = sess
        self.gradient_norm_clip_c = None
        self.gradient_norm_clip_a = None

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.G = tf.placeholder(tf.float32, [None, 2], 'G')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.D = tf.placeholder(tf.float32, [None, 1], 'd')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(tf.concat([self.S, self.G], axis=1), scope='eval', trainable=True)
            a_ = self._build_a(tf.concat([self.S_, self.G], axis=1), scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(tf.concat([self.S, self.G], axis=1), self.a, scope='eval', trainable=True)
            q_ = self._build_c(tf.concat([self.S_, self.G], axis=1), a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + (1 - self.D) * GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        a_loss = - tf.reduce_mean(q)  # maximize the q

        if self.gradient_norm_clip_c is not None:
            c_optimizer = tf.train.AdamOptimizer(LR_C)
            c_gradients = c_optimizer.compute_gradients(td_error, var_list=self.ce_params)
            for i, (grad, var) in enumerate(c_gradients):
                if grad is not None:
                    c_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm_clip_c), var)
            self.ctrain = c_optimizer.apply_gradients(c_gradients)

            if self.gradient_norm_clip_a is not None:
                a_optimizer = tf.train.AdamOptimizer(LR_A)
                a_gradients = c_optimizer.compute_gradients(a_loss, var_list=self.ae_params)
                for i, (grad, var) in enumerate(a_gradients):
                    if grad is not None:
                        a_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm_clip_a), var)
                self.atrain = a_optimizer.apply_gradients(a_gradients)
            else:
                self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        else:
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
            self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, g):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :], self.G: g[np.newaxis, :]})[0]

    def update_target_net(self):
        self.sess.run(self.soft_replace)

    def learn(self, optimization_steps):
        if len(self.memory) < BATCH_SIZE:  # if there's no enough transitions, do nothing
            return 0, 0
        # soft target replacement
        for _ in range(optimization_steps):
            minibatch = np.vstack(random.sample(self.memory, BATCH_SIZE))
            bs = np.vstack(minibatch[:, 0])
            ba = np.vstack(minibatch[:, 1])
            br = np.vstack(minibatch[:, 2])
            bs_ = np.vstack(minibatch[:, 3])
            bd = np.vstack(minibatch[:, 4])
            bg = np.vstack(minibatch[:, 5])

            self.sess.run(self.atrain, {self.S: bs, self.G: bg})
            self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.D: bd, self.G: bg})

    def store_transition(self, ep_experience):
        self.memory += ep_experience.memory
        if len(self.memory) > MEMORY_CAPACITY:
            self.memory = self.memory[-MEMORY_CAPACITY:]

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            net = tf.layers.dense(tf.concat([s, a], axis=1), 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


sess = tf.Session()

# Create actor and critic.
ddpg = DDPG(sess, ACTION_DIM, STATE_DIM, ACTION_BOUND[1])
ep_experience = Episode_experience()
ep_experience_her = Episode_experience()

saver = tf.train.Saver()
path = './' + MODE[n_model] + '_1'

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())


def train():
    var = 2.  # control exploration
    total_step = 1

    for ep in range(MAX_EPISODES):
        s, g = env.reset()
        ep_reward = 0

        for ep_step in range(MAX_EP_STEPS):
            total_step += 1
            # while True:
            if RENDER:
                env.render()

            # Added exploration noise
            a = ddpg.choose_action(s, g)
            a = np.clip(np.random.normal(a, np.max([VAR_MIN, var])), *ACTION_BOUND)  # add randomness to action selection for exploration
            s_, r, done, g = env.step(a)
            d = 1 if done else 0
            ep_experience.add(s, a, r, s_, d, g)

            s = s_
            ep_reward += r

            if ep_step == MAX_EP_STEPS - 1 or (ep_step+1) % 10 or done:
                if use_her:  # The strategy can be changed here
                    for t in range(len(ep_experience.memory)):
                        for _ in range(K):
                            future = np.random.randint(t, len(ep_experience.memory))
                            goal_ = ep_experience.memory[future][3][-2:]  # next_state of future
                            state_ = ep_experience.memory[t][0]
                            action_ = ep_experience.memory[t][1]
                            next_state_ = ep_experience.memory[t][3]
                            if np.sqrt(np.sum(np.square(200*state_[-2:] - 200*goal_))) < env.point_l:
                                reward_ = 1
                                done_ = 0

                            else:
                                reward_ = 0
                                done_ = 0
                            ep_experience_her.add(state_, action_, reward_, next_state_, done_, goal_)
                    ddpg.store_transition(ep_experience_her)
                    ep_experience_her.clear()

                if ep_step == MAX_EP_STEPS - 1 or done:
                    ddpg.store_transition(ep_experience)
                    ep_experience.clear()

                if len(ddpg.memory) >= BATCH_SIZE:
                    ddpg.learn(5)
                    ddpg.update_target_net()

            if len(ddpg.memory) >= BATCH_SIZE:# and total_step % 5 == 0:
                var *= .9995

            if ep_step == MAX_EP_STEPS - 1 or done:
                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % np.max([VAR_MIN, var]),
                      )
                break

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./' + MODE[n_model], 'DDPG_her.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)


def eval():
    for ep in range(MAX_EPISODES):
        env.set_fps(30)
        s, g = env.reset()
        while True:
            if RENDER:
                env.render()
            a = ddpg.choose_action(s, g)
            s_, r, done = env.step(a)
            s = s_



if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train()
