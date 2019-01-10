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

    def force_bot_exploitation(self):
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
        """ Method to assign rewards to actions between states. """
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

    def swap_move_selections(self, gameboard):
        """ Method to swap between exploration and exploitation modes. """
        current_state_key = TicTacToe_GameAgent.flatten_gameboard(gameboard)
        exploration = np.random.random() < self.exploration_rate
        print("explore" if exploration or current_state_key not in self.states else "exploit")
        current_action = self.initiate_explorations(gameboard) if exploration or current_state_key not in self.states else self.initiate_exploitations(current_state_key)
        print(current_action)
        self.set_state_by_action(gameboard, current_action)
        return current_action

    def initiate_explorations(self, gameboard, state_key=None):
        """ Method to initiate bot's exploration mode to locate empty cell. """
        if state_key:
            state_values = self.states[state_key]
            print("State Rewards: {}".format(state_values))
        empties_X, empties_Y = np.where(gameboard == 0)
        empty_cells = [(X, Y) for X, Y in zip(empties_X, empties_Y)]
        if len(empty_cells) > 0:
            random_empty_cell_index = np.random.choice(range(len(empty_cells)))
            return empty_cells[random_empty_cell_index]

    def initiate_exploitations(self, state_key):
        """ Method to initiate bot's exploitation mode to determine best play action. """
        state_values = self.states[state_key]
        print("State Rewards: {}".format(state_values))
        optimal_actions_X, optimal_actions_Y = np.where(state_values == state_values.max())
        optimal_value_indices = [(X, Y) for X, Y in zip(optimal_actions_X, optimal_actions_Y)]
        index_choice = np.random.choice(len(optimal_value_indices))
        return optimal_value_indices[index_choice]
