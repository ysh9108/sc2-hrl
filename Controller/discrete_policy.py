# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import copy
from common.constant import SUB_POLICIES_DICT, EMBEDDING_SIZE
from common.preprocess import spatial_minimap_data, spatial_screen_data, spatial_value_name, non_spatial_data
from common.preprocess import AgentScreenInputTuple, AgentMinimapInputTuple, AgentNonSpatialInputTuple

class PPO(object):

    def __init__(
            self,
            global_step,
            batch_size,
            state_size,
            action_size,
            hidden_layer_size,
            actor_update_steps,
            critic_update_steps,
            actor_learning_rate,
            critic_learning_rate,
            kl_penalty_target,
            kl_penalty_lam,
            clip_epsilon,
            gamma,
            epsilon_max,
            epsilon_len,
            model_name
    ):
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size

        self.hidden_layer_size = hidden_layer_size

        self.a_update_steps = actor_update_steps
        self.c_update_steps = critic_update_steps

        self.method = [
            dict(name='kl_pen', kl_target=kl_penalty_target, lam=kl_penalty_lam),
            dict(name='clip', epsilon=clip_epsilon),
        ][1]

        # Shape : None, 84, 84, n_f_dim ( one_hot 특징 맵 수 )
        self.tfs = tf.placeholder(tf.float32, [None, state_size[1], state_size[2], state_size[3]], 'state')


        # Shape : None, 84, 84, 64
        conv1 = tf.layers.conv2d(inputs=self.tfs, filters=32, activation=tf.nn.relu, kernel_size=[8, 8],
                                 strides=[1, 1], padding="SAME", name='CONV2D_1')
        # Shape : None, 84, 84, 128
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, activation=tf.nn.relu, kernel_size=[3, 3],
                                 strides=[1, 1], padding="SAME", name='CONV2D_2')

        self.flat = tf.layers.flatten(conv2)

        self.infotfs = tf.placeholder(dtype=tf.float32, shape=[None,EMBEDDING_SIZE], name='info_state')

        self.info = tf.concat([self.flat, self.infotfs], axis=1)

        # critic
        with tf.variable_scope('critic'):
            # l1 = tf.layers.dense(self.flat, self.hidden_layer_size, tf.nn.relu)
            l1 = tf.layers.dense(self.info, self.hidden_layer_size, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))

            # actor
            pi, pi_params = self._build_anet('pi', trainable=True)
            oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

            with tf.variable_scope('sample_action'):
                self.sample_op = pi

            # update_oldpi_op
            with tf.variable_scope('update_oldpi'):
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            with tf.variable_scope('train_input'):

                self.action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name='actions')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

            pi_prob = np.multiply(pi, self.action)
            pi_prob = tf.reduce_sum(pi_prob, axis=1)

            oldpi_prob = np.multiply(oldpi, self.action)
            oldpi_prob = tf.reduce_sum(oldpi_prob, axis=1)

            with tf.variable_scope('loss'):
                ratios = tf.exp(tf.log(tf.clip_by_value(pi_prob, 1e-10, 1.0)) - tf.log(tf.clip_by_value(oldpi_prob, 1e-10, 1.0)))
                clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.method['epsilon'], clip_value_max=1 + self.method['epsilon'])
                loss_clip = tf.minimum(ratios, tf.multiply(self.tfadv, clipped_ratios))
                loss_clip = tf.reduce_mean(loss_clip)

                entropy = -tf.reduce_sum(pi * tf.log(tf.clip_by_value(pi, 1e-10, 1.0)), axis=1)
                entropy = tf.reduce_mean(entropy, axis=0)
                self.aloss = loss_clip + entropy * 0.01

            with tf.variable_scope('train_op'):

                # if global_step % 20 == 0 and global_step != 0:
                #     print("Global step is {}".format(global_step))
                #     actor_learning_rate =  actor_learning_rate * gamma
                #     critic_learning_rate = critic_learning_rate * gamma
                self.closs += entropy * 0.01
                self.ctrain_op = tf.train.AdamOptimizer(critic_learning_rate).minimize(self.closs)
                self.atrain_op = tf.train.AdamOptimizer(actor_learning_rate).minimize(self.aloss, global_step=global_step)

            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

    def _build_anet(self, name, trainable):
        # discrete action
        with tf.variable_scope(name):
            # l1 = tf.layers.dense(self.flat, self.hidden_layer_size, tf.nn.relu, trainable=trainable)
            l1 = tf.layers.dense(self.info, self.hidden_layer_size,tf.nn.relu,trainable=trainable)
            action = tf.layers.dense(l1, self.action_size, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return action, params
