"""
TITLE:          GameAgent.py (TicTacToe)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Object structure containing logic for allowing the bot
                to manipulate the game board environment and win the game.
"""

class TicTacToe_GameAgent(object):
    """ Logic machine to allow bot to make decisions to play the TicTacToe game. """
    def __init__(self, exploration_rate=0.33, learning_rate=0.5, discount_factor=0.01):
        self.states = dict()
        self.state_order = list()
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor