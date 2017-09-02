import gym
import numpy as np
from policy_gradient import Agent
import tensorflow as tf


EPISODES = 1000000
env = gym.make('CartPole-v1')

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
discount = 0.99

agent = Agent(n_state, n_action, discount)
scores = []
summary_score = tf.Summary()

for e in range(EPISODES):
    done = False
    state = env.reset()
    state = np.reshape(state, [1, n_state])
    score = 0

    while not done:
        #env.render()
        action = agent.get_action(state)

        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [-1, n_state])
        agent.train(state, action, reward, next_state, done)

        print(reward)
        state = next_state

        score +=reward
    summary_score.value.add(tag='score', simple_value=score)
    agent.writer.add_summary(summary_score, e)

    scores.append(score)
    print(e, " -> ", score)
    if np.mean(scores[min(-10, len(scores)):]) > 490:
        break






