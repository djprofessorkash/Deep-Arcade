"""
TITLE:          PlayerInstance.py (Snake)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Support Python file holding the player object instance for 
                the deep learning reinforcement simulation of Snake.
"""

import pygame

class PlayerInstance(object):
    """ Object storing player instance to progress through the game session. """
    def __init__(self, game_session):
        dim_x, dim_y = 0.45 * game_session.play_width, 0.5 * game_session.play_height
        self.dim_x, self.dim_y = dim_x - (dim_x % 20), dim_y - (dim_y % 20)
        self.position = list()
        self.position.append([self.dim_x, self.dim_y])
        self.food = 1
        self.is_eaten = False
        self.image = pygame.image.load("img/SnakeBody.png")
        self.delta_x, self.delta_y = 20, 0