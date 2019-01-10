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
