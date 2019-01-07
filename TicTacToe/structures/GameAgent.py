"""
TITLE:          GameAgent.py (TicTacToe)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Object structure containing logic for allowing the bot
                to manipulate the game board environment and win the game.
"""

import numpy as np

class TicTacToe_GameAgent(object):
    """ Logic machine to allow bot to make decisions to play the TicTacToe game. """
    def __init__(self, exploration_rate=0.33, learning_rate=0.5, discount_factor=0.01):
        self.states = dict()
        self.state_order = list()
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    @staticmethod
    def flatten_gameboard(gameboard):
        """ Static method to convert gameboard grid into serialized, flattened array. """
        gameboard_serialization = gameboard.flatten()
        return "".join([str(iterator) for iterator in gameboard_serialization.flatten().tolist()])

    def initiate_exploitations(self):
        """ Method to end bot's exploration mode and begin bot's moves towards exploitation. """
        self.exploration_rate = 0

    def temporal_difference_learner(self, reward, new_state_key, old_state_key):
        """ Method to teach reinforcement model to learn via temporal difference algorithm. """
        old_state = self.states.get(old_state_key, np.zeros((3, 3)))
        return self.learning_rate * ((reward * self.states[new_state_key]) - old_state)

    def set_state_by_action(self, old_gameboard, action):
        """ Method to save action to given state across gameboard. """
        state_key = TicTacToe_GameAgent.flatten_gameboard(old_gameboard)
        self.state_order.append((state_key, action))

    def assign_rewards_to_actions(self, reward):
        """  """
        if len(self.state_order) == 0:
            return None
        new_state_key, new_action = self.state_order.pop()
        self.states[new_state_key] = np.zeros((3, 3))
        self.states[new_state_key].itemset(new_action, reward)
        while self.state_order:
            current_state_key, current_action = self.state_order.pop()
            reward *= self.discount_factor
            if current_state_key in self.states:
                reward += self.temporal_difference_learner(reward, new_state_key, current_state_key).item(new_action)
                self.states[current_state_key].itemset(current_action, reward)
            else:
                self.states[current_state_key] = np.zeros((3, 3))
                reward = self.temporal_difference_learner(reward, new_state_key, current_state_key).item(new_action)
                self.states[current_state_key].itemset(current_action, reward)
            new_state_key, new_action = current_state_key, current_action
            