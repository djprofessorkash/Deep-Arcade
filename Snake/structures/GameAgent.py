"""
TITLE:          GameAgent.py (Snake)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Support Python file holding the reinforcement learning algorithm for 
                the deep learning reinforcement simulation of Snake.
NOTE:           Uses the Deep-Q Networks Reinforcement model.
"""

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add

class GameAgent(object):
    """ Object structure storing the Deep Reinforcement Learning Agent. """
    def __init__(self):
        pass

    def get_game_state(self):
        pass

    def get_game_reward(self):
        pass

    def produce_network_architecture(self):
        pass

    def save_state_to_memory(self):
        pass

    def short_term_memory_trainer(self):
        pass

    def replay_from_memory(self):
        pass