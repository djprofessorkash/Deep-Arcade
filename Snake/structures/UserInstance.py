"""
TITLE:          UserInstance.py (Snake)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Support Python file holding the user player object instance for 
                the deep learning reinforcement simulation of Snake.
"""

class UserInstance(object):
    """ Object containing user interface for Snake game. """
    def __init__(self, speed, board_size=(440, 440)):
        dim_x, dim_y = 0.45 * board_size[0], 0.5 * board_size[1]
        self.dim_x, self.dim_y = dim_x - (dim_x % 20), dim_y - (dim_y % 20)
        self.speed = speed
        
    def moveRight(self):
        self.dim_x = self.dim_x + self.speed

    def moveLeft(self):
        self.dim_x = self.dim_x - self.speed

    def moveUp(self):
        self.dim_y = self.dim_y - self.speed

    def moveDown(self):
        self.dim_y = self.dim_y + self.speed