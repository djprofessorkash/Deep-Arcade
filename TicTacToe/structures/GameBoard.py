"""
TITLE:          GameBoard.py (TicTacToe)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Object structure containing logic for setting up 
                the Tic Tac Toe gameboard environment.
"""

import numpy as np

class Tic_Tac_Toe_GameBoard(object):
    """ Game board environment for Tic Tac Toe. """
    def __init__(self, N=3, user_play_symbol="X"):
        self.N = N
        self.gameboard = None
        self.reset_gameboard(N)
        self.stale = False
        self.syms_empty = {
            "symbol": " ",
            "value": 0
        }
        self.syms_O = {
            "symbol": "O",
            "value": 1
        }
        self.syms_X = {
            "symbol": "X",
            "value": 2
        }
        self.player_sym, self.bot_sym = (self.syms_X, self.syms_O) if user_play_symbol.upper() == "X" else (self.syms_O, self.syms_X)
        self.winner = None

    def _reset_gameboard(self):
        """ Helper method to reset and reinitialize the gameboard environment. """
        self.gameboard = np.zeros((self.N, self.N)).astype(int)
        self.winner = None