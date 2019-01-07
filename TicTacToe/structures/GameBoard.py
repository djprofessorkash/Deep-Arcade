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
            return self.syms_X.get("symbol")
        elif element == self.syms_O.get("value"):
            return self.syms_O.get("symbol")
        else:
            return self.syms_empty.get("symbol")

    def create_gameboard_architecture(self):
        """ Method that draws the game board architecture iteratively as empty grid. """
        gameboard_elements = self.gameboard.size
        play_symbols = [self.create_play_symbol_by_element(self.gameboard.item(iterator)) for iterator in range(gameboard_elements)]
        gameboard_architecture = """
             {} | {} | {}
            -----------
             {} | {} | {}
            -----------
             {} | {} | {}
        """.format(*play_symbols)
        print(gameboard_architecture)

    def _axis_has_same_values(self, axis, element, el_X, el_Y):
        """ Helper method that checks if row/column of gameboard matrix has same values throughout. """
        MAX_CEILING, resultant_ = self.gameboard.shape[0], True
        row_index, column_index = 0, 0
        iterator_major, iterator_fixed, iterator_redundant = (column_index, el_X, el_Y) if axis == 0 else (row_index, el_Y, el_X)
        while iterator_major < MAX_CEILING:
            if iterator_major != iterator_redundant:
                gameboard_element = self.gameboard[iterator_fixed][iterator_major] if axis == 0 else self.gameboard[iterator_major][iterator_fixed]
                if gameboard_element != element or gameboard_element == 0:
                    resultant_ = False
                    break
            iterator_major += 1
        return resultant_

    def _diagonal_has_same_values(self, orientation, element, el_X, el_Y):
        """ Helper method that checks if checked diagonal of gameboard matrix has same values throughout. """
        MAX_CEILING, resultant_ = self.gameboard.shape[0], True
        iterator = int()
        if orientation == "left":
            jterator = int()
        elif orientation == "right":
            jterator = MAX_CEILING - 1
        while iterator < MAX_CEILING:
            if iterator != el_X:
                if self.gameboard[iterator][jterator] != element or self.gameboard[iterator][jterator] == 0:
                    resultant_ = False
                    break
            iterator += 1
            if orientation == "left":
                jterator += 1
            elif orientation == "right":
                jterator -= 1
        return resultant_

    def _lines_have_same_values(self, orientation, element, el_X, el_Y):
        """  """
        if orientation == "columns":
            axis = 1
        elif orientation == "rows":
            axis = 0
        return self._axis_has_same_values(axis, element, el_X, el_Y)

def main():
    game = TicTacToe_GameBoard()
    game.create_gameboard_architecture()

if __name__ == "__main__":
    main()