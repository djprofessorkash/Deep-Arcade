"""
TITLE:          GameBoard.py (Snake)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Support Python file holding the game session instance for 
                the deep learning reinforcement simulation of Snake.
"""

import pygame

class GameBoard(object):
    """ Object structure storing the GameBoard session. """
    def __init__(self, play_width, play_height):
        pygame.display.set_caption("SnakeGen")
        self.play_width = play_width
        self.play_height = play_height
        self.play_display = pygame.display.set_mode((play_width, play_height + 60))
        # self.background = pygame.image.load("img/background.png")
        self.has_crashed = False
        self.player = PlayerInstance(self)
        self.food = PelletInstance()
        self.score = 0