"""
TITLE:          main.py (Tic Tac Toe)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Main Python file for controlling deep learning reinforcement simulation 
                of Tic Tac Toe.
"""

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import seaborn as sns

class Tic_Tac_Toe_Environment(object):
    """ Game board environment for Tic Tac Toe. """
    def __init__(self, N=3, user_play_symbol="X"):
        self.N = N
        self.gameboard = None
        self.reset_gameboard(N)
        self.stale = False
        self.syms_empty = {
            "mark": " ",
            "value": 0
        }
        self.syms_O = {
            "mark": "O",
            "value": 1
        }
        self.syms_X = {
            "mark": "X",
            "value": 2
        }
        self.player_sym, self.bot_sym = (self.syms_X, self.syms_O) if user_play_symbol.upper() == "X" else (self.syms_O, self.syms_X)
        self.winner = None

def main():
    """ Main run function. """
    pass

if __name__ == "__main__":
    main()