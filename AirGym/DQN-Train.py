# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:16:10 2017

@author: Kjell
"""
import numpy as np
import gym

import envs.airsim

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from PIL import Image

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


ENV_NAME = "AirSimMultirotorEnv-v6"

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


train = True

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)


# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.2, value_test=.05,
                                  nb_steps=100000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1500,
               target_model_update=1e-2, policy=policy, gamma=.97)

dqn.compile(Adam(lr=0.00025), metrics=['mae'])


if train:
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    
    
    log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
    callbacks = [FileLogger(log_filename, interval=10)]

    dqn.fit(env, callbacks=callbacks, nb_steps=250000, visualize=False, verbose=2)
    
    
    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    
    # Finally, evaluate our algorithm for 5 episodes.
    
    #dqn.test(env, nb_episodes=5, visualize=False)

else:

    dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
    dqn.test(env, nb_episodes=10, visualize=False)



