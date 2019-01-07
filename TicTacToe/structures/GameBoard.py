"""
TITLE:          GameBoard.py (TicTacToe)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Object structure containing logic for setting up 
                the Tic Tac Toe gameboard environment.
"""

import numpy as np

class TicTacToe_GameBoard(object):
    """ Game board environment for Tic Tac Toe. """
    def __init__(self, N=3, user_play_symbol="X"):
        self.N = N
        self.gameboard = None
        self._reset_gameboard()
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

    def create_play_symbol_by_element(self, element):
        """ Method creates current symbol (" ", "X", "O") based on current game board element position. """
        if element == self.syms_X.get("value"):
            return self.syms_X.get("mark")
        elif element == self.syms_O.get("value"):
            return self.syms_O.get("mark")
        else:
            return self.syms_empty.get("mark")

    def create_gameboard_architecture(self):
        """ Draws the game board architecture iteratively. """
        gameboard_elements = self.gameboard.size
        print(gameboard_elements)
        play_symbols = [self.create_play_symbol_by_element(self.gameboard.item(iterator) for iterator in range(gameboard_elements))]
        print(play_symbols)
        gameboard_architecture = """
             {} | {} | {}
            -----------
             {} | {} | {}
            -----------
             {} | {} | {}
        """.format(*play_symbols)
        print(gameboard_architecture)

def main():
    game = TicTacToe_GameBoard()
    game.create_gameboard_architecture()

if __name__ == "__main__":
    main()