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
from random import sample
import numpy as np
import pandas as pd
from operator import add

class GameAgent(object):
    """ Object structure storing the Deep Reinforcement Learning Agent. """
    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.df = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_prediction = 0
        self.learning_rate = 0.0005
        self.model = self.produce_network_architecture()
        self.epsilon = 0
        self.actual = list()
        self.memory = list()

    def get_game_state(self, game_session, player_instance, pellet_instance):
        """ Method to logically determine detailed current game state using game session parameters. """
        move_right = player_instance.delta_x == 20
        move_left = player_instance.delta_x == -20
        move_up = player_instance.delta_y == -20
        move_down = player_instance.delta_y == 20

        no_move_x = player_instance.delta_x == 0
        no_move_y = player_instance.delta_y == 0

        cum_move_right = (list(map(add, player_instance.position[-1], [20, 0])) in player_instance.position)
        cum_move_left = (list(map(add, player_instance.position[-1], [-20, 0])) in player_instance.position)
        cum_move_up = (list(map(add, player_instance.position[-1], [0, -20])) in player_instance.position)
        cum_move_down = (list(map(add, player_instance.position[-1], [0, 20])) in player_instance.position)

        at_right_wall = player_instance.position[-1][0] + 20 >= (game_session.play_width - 20)
        at_left_wall = player_instance.position[-1][0] - 20 < 20
        at_top_wall = player_instance.position[-1][-1] - 20 < 20
        at_bottom_wall = player_instance.position[-1][-1] + 20 >= (game_session.play_height - 20)

        def _get_dangerous_straight_logic():
            """ Helper method to construct logical calculation of danger-straight movement. """
            risky_rights = (move_right and no_move_y and (cum_move_right or at_right_wall))
            risky_lefts = (move_left and no_move_y and (cum_move_left or at_left_wall))
            risky_ups = (no_move_x and move_up and (cum_move_up or at_top_wall))
            risky_downs = (no_move_x and move_down and (cum_move_down or at_bottom_wall))
            return risky_rights or risky_lefts or risky_ups or risky_downs

        def _get_dangerous_right_logic():
            """ Helper method to construct logical calculation of danger-right movement. """
            risky_rights = (no_move_x and move_up and (cum_move_right or at_right_wall))
            risky_lefts = (no_move_x and move_down and (cum_move_left or at_left_wall))
            risky_ups = (move_left and no_move_y and (cum_move_up or at_top_wall))
            risky_downs = (move_right and no_move_y and (cum_move_down or at_bottom_wall))
            return risky_rights or risky_lefts or risky_ups or risky_downs

        def _get_dangerous_left_logic():
            """ Helper method to construct logical calculation of danger-left movement. """
            risky_rights = (no_move_x and move_down and (cum_move_right or at_right_wall))
            risky_lefts = (no_move_x and move_up and (cum_move_left or at_left_wall))
            risky_ups = (move_right and no_move_y and (cum_move_up or at_top_wall))
            risky_downs = (move_left and no_move_y and (cum_move_down or at_bottom_wall))
            return risky_rights or risky_lefts or risky_ups or risky_downs

        STATE_VECTOR = [
            _get_dangerous_straight_logic(),       # DANGEROUS STRAIGHT
            _get_dangerous_right_logic(),          # DANGEROUS RIGHT
            _get_dangerous_left_logic(),           # DANGEROUS LEFT
            player_instance.delta_x == -20,                                     # PLAYER LEFT
            player_instance.delta_x == 20,                                      # PLAYER RIGHT
            player_instance.delta_y == -20,                                     # PLAYER UP
            player_instance.delta_y == 20,                                      # PLAYER DOWN
            pellet_instance.dim_x < player_instance.dim_x,                      # PELLET LEFT
            pellet_instance.dim_x > player_instance.dim_x,                      # PELLET RIGHT
            pellet_instance.dim_y < player_instance.dim_y,                      # PELLET UP
            pellet_instance.dim_y > player_instance.dim_y                       # PELLET DOWN
        ]
        for iterator in range(len(STATE_VECTOR)):
            if STATE_VECTOR[iterator]:
                STATE_VECTOR[iterator] = 1
            else:
                STATE_VECTOR[iterator] = 0
        return np.asarray(STATE_VECTOR)

    def get_game_reward(self, player_instance, has_crashed):
        """ Method to determine effective weights to reward or punish player activity. """
        self.reward = 0
        if has_crashed:
            self.reward = -10
            return self.reward
        if player_instance.has_eaten:
            self.reward = 10
        return self.reward

    def produce_network_architecture(self):
        """ Method to create neural network architecture using optimized Keras models. """
        model = Sequential()
        model.add(Dense(120, activation="relu", input_shape=(11,)))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation="relu"))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation="relu"))
        model.add(Dropout(0.15))
        model.add(Dense(3, activation="softmax"))
        optimizer = Adam(self.learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def save_state_to_memory(self, current_state, current_action, current_reward, next_state, stop):
        """ Method to save current detailed state to object's memory. """
        self.memory.append((current_state, current_action, current_reward, next_state, stop))

    def short_term_memory_trainer(self, current_state, current_action, current_reward, next_state, stop):
        """ Method to train short-term memory bank using detailed saved state info. """
        target_ = current_reward
        if not stop:
            target_ = current_reward + (self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0]))
        target_final = self.model.predict(current_state.reshape((1, 11)))
        target_final[0][np.argmax(current_action)] = target_
        self.model.fit(current_state.reshape((1, 11)), target_final, epochs=1, verbose=0)

    def replay_from_memory(self, memory_bank):
        """ Method to retrieve state details from object's memory bank. """
        if len(memory_bank) > 1000:
            batch = sample(memory_bank, 1000)
        else:
            batch = memory_bank
        for current_state, current_action, current_reward, next_state, stop in batch:
            target_ = current_reward
            if not stop:
                target_ = current_reward + (self.gamma * np.amax(self.model.predict(np.array([next_state]))[0]))
            target_final = self.model.predict(np.array([current_state]))
            target_final[0][np.argmax(current_action)] = target_
            self.model.fit(np.array([current_state]), target_final, epochs=1, verbose=0)
