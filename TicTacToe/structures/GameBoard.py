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
        self.is_filled = False
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
        """ Helper method that checks if current diagonal of gameboard matrix has same values throughout. """
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
        """ Helper method that checks if collective rows/columns have same values throughout game matrix.  """
        if orientation == "columns":
            axis = 1
        elif orientation == "rows":
            axis = 0
        return self._axis_has_same_values(axis, element, el_X, el_Y)

    def check_any_diagonals_have_same_values(self, element, el_X, el_Y):
        """ Method that checks if any diagonals across game matrix have same values. """
        MAX_CEILING = self.gameboard.shape[0]
        if el_X == el_Y and el_X + el_Y == MAX_CEILING - 1:
            return self._diagonal_has_same_values("left", element, el_X, el_Y) or self._diagonal_has_same_values("right", element, el_X, el_Y)
        if el_X == el_Y:
            return self._diagonal_has_same_values("left", element, el_X, el_Y)
        if el_X + el_Y == MAX_CEILING - 1:
            return self._diagonal_has_same_values("right", element, el_X, el_Y)
        else:
            return False

    def check_game_over(self, player, element, el_X, el_Y):
        """ Method that checks current game status and reports if game is over. """
        return self._lines_have_same_values("columns", element, el_X, el_Y) or self._lines_have_same_values("rows", element, el_X, el_Y) or self.check_any_diagonals_have_same_values(element, el_X, el_Y)

    def check_winning_move(self, player, element, el_X, el_Y):
        """ Method that checks if current move iteration is player's winning move. """
        if self.check_game_over(player, element, el_X, el_Y):
            self.winner = player
            return True
        else:
            return False

    def check_if_filled(self):
        """ Method that checks if entire game board is filled with moves. """
        x, y = np.where(self.gameboard == 0)
        if len(x) == 0 and len(y) == 0:
            self.is_filled = True
        return self.is_filled

    def _player_mover(self, element_symbol, el_X, el_Y):
        """ Helper method that facilitates logic to move player symbol into data-driven location. """
        current_symbol = None
        if element_symbol == self.syms_O.get("symbol"):
            current_symbol = self.syms_O
        elif element_symbol == self.syms_X.get("symbol"):
            current_symbol = self.syms_X
        else:
            return
        if self.gameboard[el_X][el_Y] == 0:
            self.gameboard[el_X][el_Y] = current_symbol.get("value")
            self.create_gameboard_architecture()
            if self.check_winning_move(current_symbol.get("symbol"), current_symbol.get("value"), el_X, el_Y):
                print("Winner is: {}!".format(self.winner))
                return self.winner
            elif self.check_if_filled():
                print("Draw!")
                return "draw"

    def play_move(self, player, el_X, el_Y):
        """ Method that allows for player-driven moves across gameboard grid. """
        MAX_CEILING = self.gameboard.shape[0]
        if el_X > MAX_CEILING - 1 or el_Y > MAX_CEILING:
            return
        if player == "user":
            self._player_mover(self.player_sym.get("symbol"), el_X, el_Y)
        elif player == "bot":
            self._player_mover(self.bot_sym.get("symbol"), el_X, el_Y)

def main():
    game = TicTacToe_GameBoard()
    game.create_gameboard_architecture()

if __name__ == "__main__":
    main()