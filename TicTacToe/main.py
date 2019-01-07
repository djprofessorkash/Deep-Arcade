"""
TITLE:          main.py (Tic Tac Toe)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Main Python file for controlling deep learning reinforcement simulation 
                of Tic Tac Toe.
"""

# TODO: Improve at architectural actualization by importing only essential libraries/functions
# from matplotlib import pyplot as plt
# from matplotlib import ticker
# import numpy as np
# import pandas as pd
# import seaborn as sns
from structures import GameBoard, GameAgent
from numpy import zeros
from pandas import DataFrame as df

def train_reinforcer(num_epochs, bot_1, bot_2, bot_sym_1, bot_sym_2):
    """ Global function to run generational training for the game's reinforcement model. """
    bot_1_wins, bot_2_wins = int(), int()
    traced_wins = df(data=zeros((num_epochs, 2)), columns=["bot_1", "bot_2"])
    for iterator in range(num_epochs):
        print("-" * 100)
        print("EPOCH: {}".format(iterator + 1))
        gameboard = GameBoard.TicTacToe_GameBoard()
        while not gameboard.is_filled:
            winner = gameboard._player_mover(bot_sym_2, *bot_2.swap_move_selections(gameboard.gameboard))
            if winner:
                _bot_optimizer(gameboard, bot_1, bot_2, bot_sym_1, bot_sym_2)
                bot_2_wins += 1
                traced_wins.set_value(iterator, "bot_2", 1)
                break
                traced_wins[iterator] = 2
            elif winner == "draw":
                break
    return traced_wins, bot_1_wins, bot_2_wins

def _bot_optimizer(game_session, bot_1, bot_2, bot_sym_1, bot_sym_2):
    """ Global helper function to optimize bot's decision vectors based on successful game outcome. """
    if game_session.winner == bot_sym_1:
        bot_1.assign_rewards_to_actions(1)
        bot_2.assign_rewards_to_actions(-1)
    elif game_session.winner == bot_sym_2:
        bot_1.assign_rewards_to_actions(-1)
        bot_2.assign_rewards_to_actions(1)

def main():
    """ Main run function. """
    bot_sym_1, bot_sym_2 = "O", "X"
    bot_1, bot_2 = GameAgent.TicTacToe_GameAgent(), GameAgent.TicTacToe_GameAgent()
    epochs_ = 5000
    traced_wins, bot_1_wins, bot_2_wins = train_reinforcer(epochs_, bot_1, bot_2, bot_sym_1, bot_sym_2)

if __name__ == "__main__":
    main()