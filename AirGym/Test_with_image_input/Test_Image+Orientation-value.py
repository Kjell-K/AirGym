# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:27:35 2017

@author: Kjell
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:16:10 2017

@author: Kjell
"""
import numpy as np
import gym

import envs.airsim

import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from PIL import Image

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from rl.callbacks import TrainEpisodeLogger, TestLogger

from keras.callbacks import History

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='AirSimMultirotorEnv-v5')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


# Next, we build our model. We use the same model that was described by Mnih et al. (2015).

INPUT_SHAPE = (119, 214)
WINDOW_LENGTH = 1

img_input = (WINDOW_LENGTH,) + INPUT_SHAPE

img = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=img_input)
img = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', input_shape=img)
img = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', input_shape=img)
img = Flatten(input_shape=img)
img = Dense(512, activation='relu', input_shape=img)


value_input = (1,2)
value = Flatten()(value_input)
value = Dense(16, activation='relu')(value)
value = Dense(16, activation='relu')(value)
value = Dense(16, activation='relu')(value)

actions = Dense(nb_actions, activation='linear')(img)(value)


model = Model([img_input, value_input], [actions])

print(model.summary())

"""
train = True


if train:

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=40000, window_length=1)
    

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.2, value_test=.05,
                                  nb_steps=20000)
    
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1500,
                   target_model_update=1e-2, policy=policy, gamma=.98)
    
    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [FileLogger(log_filename, interval=1)]

    dqn.fit(env, callbacks=callbacks, nb_steps=50000, visualize=False, verbose=2)
    
    
    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(args.env_name), overwrite=True)
    
    # Finally, evaluate our algorithm for 5 episodes.
    
    #dqn.test(env, nb_episodes=5, visualize=False)

else:
    dqn.load_weights('dqn_{}_weights.h5f'.format(args.env_name))
    dqn.test(env, nb_episodes=10, visualize=False)


"""
