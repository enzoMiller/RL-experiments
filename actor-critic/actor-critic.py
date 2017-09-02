import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

class Network:
    def __init__(self, n_state, n_action):

    # placeholders
        self.state = tf.placeholder("float", [None, n_state], name="state")
        self.advantage = tf.placeholder("float", [None, n_action], name="advantage")
        self.target_value = tf.placeholder("float", [None, 1], name="target_value")

    # Build Actor
        with tf.name_scope("actor"):
            self.h1_actor = layers.fully_connected(inputs=self.state, num_outputs=24, activation_fn=tf.nn.relu, scope="h1_actor",
                                                   weights_initializer=layers.xavier_initializer())
            self.output_actor = layers.fully_connected(inputs=self.h1_actor, num_outputs=n_action,
                                                       activation_fn=tf.nn.softmax, scope="output_actor",
                                                       weights_initializer=layers.xavier_initializer())

        with tf.name_scope("actor_metrics"):
            self.entropy = -tf.reduce_sum(tf.multiply(tf.log(self.output_actor), self.output_actor))
            tf.summary.scalar("actor_entropy", self.entropy)

        with tf.name_scope("actor_loss"):
            self.actor_loss = -tf.reduce_sum(tf.multiply(self.output_actor, self.advantage)) - 0.001 * tf.reduce_sum(self.entropy)
            tf.summary.scalar("actor_loss", self.actor_loss)
            self.trainActor_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.actor_loss)

    # Build Critic
        with tf.name_scope("critic"):
            self.h1_critic = layers.fully_connected(inputs=self.state,num_outputs=24, activation_fn=tf.nn.relu, scope="h1_critic",
                                                    weights_initializer=layers.xavier_initializer())
            self.output_critic = layers.fully_connected(inputs=self.h1_critic, num_outputs=1, activation_fn=None, scope="output_critic",
                                                        weights_initializer=layers.xavier_initializer())

        with tf.name_scope("critic_loss"):
            self.critic_loss = tf.reduce_sum(tf.squared_difference(self.target_value, self.output_critic))
            tf.summary.scalar("critic_loss", self.critic_loss)
            self.trainCritic_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.critic_loss)


class Agent:
    def __init__(self, n_state, n_action, discount):
        self.n_train = 0
        self.discount = discount
        self.network = Network(n_state, n_action)

        self.n_action = n_action
        self.init_network()

    def init_network(self):
        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.writer = tf.summary.FileWriter("./logs/nn_logs", self.sess.graph)
        self.merged = tf.summary.merge_all()

    def get_action(self, state):
        action_probs = self.sess.run(self.network.output_actor, feed_dict={self.network.state: state})[0]
        return np.random.choice(self.n_action, 1, p=action_probs)[0]

    def compute_advantage(self, state, action, reward, next_state, done):
        #one hot encoding of action :
        action = np.eye(self.n_action)[action]
        value_state = self.sess.run(self.network.output_critic,
                                    feed_dict={self.network.state: state})

        if done:
            value_nextstate = 0
        else:
            value_nextstate = self.sess.run(self.network.output_critic,
                                            feed_dict={self.network.state: next_state})

        target_value = np.reshape(reward + self.discount * value_nextstate, [-1, 1])
        stateAction_advantages = action * (reward + self.discount * value_nextstate - value_state)
        
        return stateAction_advantages, target_value

    def train(self, state, action, reward, next_state, done):
        self.n_train += 1
        advantage, target_value = self.compute_advantage(state, action, reward, next_state, done)

        _, _, summary = self.sess.run([self.network.trainActor_op, self.network.trainCritic_op, self.merged],
                                      feed_dict={
                                          self.network.state: state,
                                          self.network.advantage: advantage,
                                          self.network.target_value: target_value
                                      })

        self.writer.add_summary(summary, self.n_train)
