import tensorflow as tf
import numpy as np
import os
import shutil
import random

import matplotlib.pyplot as plt
from car_env_ver2 import CarEnv

np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)

MAX_EPISODES = 1000
MAX_EP_STEPS = 600
LR_A = 1e-3  # learning rate for actor
GAMMA = 0.99  # reward discount
TARGET_UPDATE = 25
MEMORY_SIZE = 10000
BATCH_SIZE = 64
NUM_Q = 50
RENDER = False
Train = False
LOAD = True
DISCRETE_ACTION = True
TAU = 0.01

env = CarEnv(discrete_action=DISCRETE_ACTION)
STATE_DIM = env.state_dim
HISTORICAL_WINDOW = 3
ACTION_DIM = 3


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.001
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


def plot_cdf(q_dist):
    y = np.ones((1, 32))
    y = np.array([[-1, 0, 1]]).T * y
    plt.ylim([-2, 2])
    # plt.xlim([np.max(actions_value)-5, np.max(actions_value)+5])
    plt.xlabel('Q')
    plt.plot(np.transpose(np.sort(q_dist[0], axis=1)), np.transpose(y), 'o')
    plt.draw()
    plt.pause(0.00001)
    plt.clf()


class IQNAgent:
    def __init__(self, sess, a_dim, s_dim,
                 num_tau=32,
                 num_tau_prime=8,
                 learning_rate=1e-3,
                 gamma=0.99,
                 batch_size=32,
                 buffer_size=10000,
                 target_update_step=25,
                 eta=0.1,
                 gradient_norm=None
                 ):
        self.memory = Memory(buffer_size)
        self.iter = 0
        self.sess = sess
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.buffer_size = buffer_size
        self.num_tau = num_tau
        self.num_tau_prime = num_tau_prime
        self.target_update_step = target_update_step
        self.gradient_norm = gradient_norm
        self.eta = eta

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.A = tf.placeholder(tf.float32, [None, 1], 'a')
        self.A_ = tf.placeholder(tf.float32, [None, 1], 'a')
        self.T = tf.placeholder(tf.float32, [None, None], 'theta_t')
        self.tau = tf.placeholder(tf.float32, [None, None], 'tau')
        self.tau_ = tf.placeholder(tf.float32, [None, None], 'tau_')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        self.q_theta_eval_train, self.q_mean_eval_train, self.q_theta_eval_test, self.q_mean_eval_test = self._build_net(
            self.S, self.tau,
            scope='eval_params',
            trainable=True)
        self.q_theta_next_train, self.q_mean_next_train, _, _ = self._build_net(self.S_, self.tau_,
                                                                                scope='target_params',
                                                                                trainable=False)

        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_params')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_params')

        self.params_replace = [tf.assign(qt, qe) for qt, qe in zip(self.qt_params, self.qe_params)]

        a_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.squeeze(tf.to_int32(self.A))], axis=1)
        self.q_theta_eval_a = tf.gather_nd(params=self.q_theta_eval_train, indices=a_indices)
        a_next_max_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32),
                                       tf.squeeze(tf.to_int32(self.A_))], axis=1)
        self.q_theta_next_a = tf.gather_nd(params=self.q_theta_next_train, indices=a_next_max_indices)

        self.loss, self.qr_error = self.quantile_huber_loss()

        if self.gradient_norm is not None:
            q_optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.01 / self.batch_size)
            q_gradients = q_optimizer.compute_gradients(self.loss, var_list=self.qe_params)
            for i, (grad, var) in enumerate(q_gradients):
                if grad is not None:
                    q_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm), var)
            self.train_op = q_optimizer.apply_gradients(q_gradients)
        else:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.01 / self.batch_size).minimize(
                self.loss, var_list=self.qe_params)

        self.saver = tf.train.Saver()

    def _build_net(self, s, tau, scope, trainable):

        s_tiled = tf.tile(s, [1, tf.shape(tau)[1]])
        s_reshaped = tf.reshape(s_tiled, [-1, self.s_dim * HISTORICAL_WINDOW])
        tau_reshaped = tf.reshape(tau, [-1, 1])

        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            pi_mtx = tf.constant(np.expand_dims(np.pi * np.arange(0, 64), axis=0), dtype=tf.float32)

            def _noisy_dense(X, n_input, n_output, name_layer, trainable):
                W_mu = tf.get_variable("W_mu_" + name_layer, shape=[n_input, n_output],
                                       initializer=tf.random_uniform_initializer(-tf.sqrt(3 / n_input),
                                                                                 tf.sqrt(3 / n_input)),
                                       trainable=trainable)

                W_sig = tf.get_variable("W_sig_" + name_layer, shape=[n_input, n_output],
                                        initializer=tf.constant_initializer(0.017), trainable=trainable)

                B_mu = tf.get_variable("B_mu_" + name_layer, shape=[1, n_output],
                                       initializer=tf.random_uniform_initializer(-tf.sqrt(3 / n_input),
                                                                                 tf.sqrt(3 / n_input)),
                                       trainable=trainable)

                B_sig = tf.get_variable("B_sig_" + name_layer, shape=[1, n_output],
                                        initializer=tf.constant_initializer(0.017), trainable=trainable)

                W_fc = tf.add(W_mu, tf.multiply(W_sig, tf.random_normal(shape=[n_input, n_output])))

                B_fc = tf.add(B_mu, tf.multiply(B_sig, tf.random_normal(shape=[1, n_output])))

                pre_noisy_layer = tf.add(tf.matmul(X, W_fc), B_fc)

                return pre_noisy_layer

            s_img = tf.reshape(s_reshaped, [-1, STATE_DIM, HISTORICAL_WINDOW, 1])
            init_wc = tf.contrib.layers.xavier_initializer_conv2d()
            c_out = tf.layers.conv2d(s_img, 10, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                     kernel_initializer=init_wc,
                                     name='l1',
                                     trainable=trainable)
            net_psi = tf.contrib.layers.flatten(c_out)

            cos_tau = tf.cos(tf.matmul(tau_reshaped, pi_mtx))
            net_phi = tf.layers.dense(cos_tau, 270, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='phi',
                                      trainable=trainable)

            joint_term = net_psi + tf.multiply(net_psi, net_phi)

            net = tf.nn.selu(_noisy_dense(joint_term, 270, 128, "layer1", trainable=trainable))

            q_flat = _noisy_dense(net, 128, self.a_dim, "layer3", trainable=trainable)

            q_re_train = tf.transpose(tf.split(q_flat, self.batch_size, axis=0), perm=[0, 2, 1])

            q_re_test = tf.transpose(tf.split(q_flat, 1, axis=0), perm=[0, 2, 1])

            q_mean_train = tf.reduce_mean(q_re_train, axis=2)

            q_mean_test = tf.reduce_mean(q_re_test, axis=2)

        return q_re_train, q_mean_train, q_re_test, q_mean_test

    def update_target_net(self):
        self.sess.run(self.params_replace)

    def quantile_huber_loss(self):
        q_theta_expand = tf.tile(tf.expand_dims(self.q_theta_eval_a, axis=2), [1, 1, self.num_tau_prime])
        T_theta_expand = tf.tile(tf.expand_dims(self.T, axis=1), [1, self.num_tau_prime, 1])

        u_theta = T_theta_expand - q_theta_expand

        rho_u_tau = self._rho_tau(u_theta, tf.tile(tf.expand_dims(self.tau, axis=2), [1, 1, self.num_tau_prime]))

        qr_error = tf.reduce_sum(tf.reduce_mean(rho_u_tau, axis=2), axis=1)

        qr_loss = tf.reduce_sum(self.ISWeights * qr_error)

        return qr_loss, qr_error

    def memory_add(self, s, a, r, s_, d):

        tau = np.random.rand(self.batch_size, self.num_tau_prime)
        tau_ = np.random.rand(self.batch_size, self.num_tau_prime)
        tau_beta_ = self.conditional_value_at_risk(self.eta, np.random.rand(self.batch_size, self.num_tau))

        T_mean_K = self.sess.run(self.q_mean_next_train, feed_dict={self.S_: np.repeat(s_[np.newaxis, :],
                                                                                     self.batch_size,
                                                                                     0),
                                                                    self.tau_: tau_beta_})
        ba_ = np.expand_dims(np.argmax(T_mean_K, axis=1), axis=1)

        T_theta_ = self.sess.run(self.q_theta_next_a, feed_dict={self.S_: np.repeat(s_[np.newaxis, :],
                                                                                     self.batch_size,
                                                                                     0),
                                                                 self.A_: ba_,
                                                                 self.tau_: tau_})

        T_theta = np.repeat(np.copy(r)[np.newaxis, :], self.batch_size, 0) + \
                  (1 - np.repeat(np.copy(d)[np.newaxis, :], self.batch_size, 0)) * self.gamma * T_theta_
        T_theta = T_theta.astype(np.float32)

        error = self.sess.run(self.qr_error, feed_dict={self.S: np.repeat(s[np.newaxis, :], self.batch_size, 0),
                                                        self.A: np.repeat(np.copy(a)[np.newaxis, :], self.batch_size,
                                                                          0),
                                                        self.T: T_theta,
                                                        self.tau: tau})

        self.memory.add(error[0], (s, a, r, s_, d))

    def learn(self):
        if self.iter % self.target_update_step == 0:
            self.update_target_net()

        minibatch, idxs, IS_weight = self.memory.sample(self.batch_size)
        minibatch = np.array(minibatch)
        bs = np.vstack(minibatch[:, 0])
        ba = np.vstack(minibatch[:, 1])
        br = np.vstack(minibatch[:, 2])
        bs_ = np.vstack(minibatch[:, 3])
        bd = np.vstack(minibatch[:, 4])
        IS_weight = IS_weight[:, np.newaxis]

        tau = np.random.rand(self.batch_size, self.num_tau_prime)
        tau_ = np.random.rand(self.batch_size, self.num_tau_prime)
        tau_beta_ = self.conditional_value_at_risk(self.eta, np.random.rand(self.batch_size, self.num_tau))

        T_mean_K = self.sess.run(self.q_mean_next_train, feed_dict={self.S_: bs_, self.tau_: tau_beta_})
        ba_ = np.expand_dims(np.argmax(T_mean_K, axis=1), axis=1)

        T_theta_ = self.sess.run(self.q_theta_next_a, feed_dict={self.S_: bs_, self.A_: ba_, self.tau_: tau_})

        T_theta = br + (1 - bd) * self.gamma * T_theta_
        T_theta = T_theta.astype(np.float32)

        loss, qr_error, _ = self.sess.run([self.loss, self.qr_error, self.train_op], {self.S: bs,
                                                                                      self.A: ba,
                                                                                      self.T: T_theta,
                                                                                      self.tau: tau,
                                                                                      self.ISWeights: IS_weight})

        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, qr_error[i])

        self.iter += 1

        return loss

    @staticmethod
    def conditional_value_at_risk(eta, tau):
        return eta * tau

    @staticmethod
    def _rho_tau(u, tau, kappa=1):
        delta = tf.cast(u < 0, 'float')
        if kappa == 0:
            return (tau - delta) * u
        else:
            return tf.abs(tau - delta) * tf.where(tf.abs(u) <= kappa, 0.5 * tf.square(u),
                                                  kappa * (tf.abs(u) - kappa / 2))

    def choose_action(self, state):
        state = state[np.newaxis, :]
        tau_K = np.random.rand(1, self.num_tau)
        tau_beta = self.conditional_value_at_risk(self.eta, tau_K)

        actions_value, q_dist = self.sess.run([self.q_mean_eval_test, self.q_theta_eval_test],
                                              feed_dict={self.S: state, self.tau: tau_beta})
        action = np.argmax(actions_value)

        return action, actions_value, q_dist


sess = tf.Session()

agent = IQNAgent(sess, ACTION_DIM, HISTORICAL_WINDOW * STATE_DIM,
                   learning_rate=LR_A,
                   gamma=GAMMA,
                   batch_size=BATCH_SIZE,
                   buffer_size=MEMORY_SIZE,
                   target_update_step=TARGET_UPDATE,
                   )

saver = tf.train.Saver()
path = './IQN_discrete' if DISCRETE_ACTION else './continuous2'

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())


def train():
    steps_last_20_ep = []
    for ep in range(MAX_EPISODES):
        s = env.reset()
        s_t = s[:, np.newaxis, np.newaxis]

        for _ in range(HISTORICAL_WINDOW - 1):
            s_t = np.append(s_t, s[:, np.newaxis, np.newaxis], axis=1)

        ep_step = 0

        for t in range(MAX_EP_STEPS):
            # while True:
            if RENDER:
                env.render()

            # Added exploration noise
            a, _, _ = agent.choose_action(s_t.reshape(1, -1)[0])
            s_, r, done = env.step(a)
            s_t_ = np.append(s_t[:, 1:, :], s_[:, np.newaxis, np.newaxis], axis=1)

            d = 1 if done else 0

            agent.memory_add(s_t.reshape(1, -1)[0],
                             [a],
                             [r],
                             s_t_.reshape(1, -1)[0],
                             [d])

            if agent.memory.tree.n_entries >= 1000:
                agent.learn()

            s = s_
            s_t = s_t_

            ep_step += 1

            if done or t == MAX_EP_STEPS - 1:
                # if done:
                print('Ep:', ep,
                      '| Steps: %i' % int(ep_step),
                      '| Reward: %.2f' % r,
                      )

                if len(steps_last_20_ep) == 20:
                    steps_last_20_ep = np.append(steps_last_20_ep[1:], r)
                else:
                    steps_last_20_ep = np.append(steps_last_20_ep, r)

                break

        if (ep + 1) % 100 == 0 or (np.mean(steps_last_20_ep) >= 1 and len(steps_last_20_ep) == 20):
            if not os.path.isdir(path): os.mkdir(path)
            ckpt_path = os.path.join(path, 'IQN.ckpt')
            save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
            print("\nSave Model %s\n" % save_path)

            if np.mean(steps_last_20_ep) >= 1 and len(steps_last_20_ep) == 20:
                break


def eval():
    # env.set_fps(10)
    ep = 1

    while True:
        s = env.reset()
        s_t = s[:, np.newaxis, np.newaxis]

        for _ in range(HISTORICAL_WINDOW - 1):
            s_t = np.append(s_t, s[:, np.newaxis, np.newaxis], axis=1)

        t = 0
        done = False

        while not done:
            env.render()
            a, _, q_dist = agent.choose_action(s_t.reshape(1, -1)[0])
            s_, r, done = env.step(a)
            s_ = np.minimum(1, s_ + 0.2 * np.random.rand(STATE_DIM) - 0.1)
            s_t_ = np.append(s_t[:, 1:, :], s_[:, np.newaxis, np.newaxis], axis=1)

            #plot_cdf(q_dist)

            s = s_
            s_t = s_t_

            t += 1

            if done:
                # if done:
                print('Ep:', ep,
                      '| Steps: %i' % int(t)
                      )

        ep += 1


if __name__ == '__main__':
    if Train:
        train()
    else:
        eval()
