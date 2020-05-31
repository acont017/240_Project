import tensorflow as tf

import time
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrappers import wrapper

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper(env)

states = (84, 84, 4)
actions = env.action_space.n


def predict(state):
        #print(session.run(tf.compat.v1.trainable_variables('target')))
        #kernel, bias = session.run(tf.compat.v1.trainable_variables('target'))[-2:]
        acts = (session.run(fetches='target/StopGradient:0', feed_dict={'input:0': np.array(state)}))
        #print(acts.shape)
        #print(acts[iter,:])
        return acts


def run(state):
    # Policy action
    q = predict(np.array(np.expand_dims(state, 0)))
    action = np.argmax(q)
    return action





tf.compat.v1.disable_eager_execution()
session = tf.compat.v1.Session()
saver = tf.compat.v1.train.import_meta_graph('./models/model-1500000.meta')
for tensor in tf.compat.v1.get_default_graph().get_operations()[:-1600]:
    print (tensor.name)
graph = tf.compat.v1.get_default_graph()
saver.restore(session, './models/model-1500000')
print(tf.compat.v1.trainable_variables('target'))


#print(graph.get_tensor_by_name('actions:0'))

# Reward
total_reward = 0
iter = 0

state = env.reset()



while True:

    env.render()

    # Run agent
    action = run(state)

    # Perform action
    next_state, reward, done, info = env.step(action=action)
    print(session.run(fetches='target/StopGradient:0', feed_dict={'input:0': np.array(np.expand_dims(state, 0))}))
    # Update state
    state = next_state

    # Increment
    iter += 1

    # If done break loop
    if done or info['flag_get']:
        break


