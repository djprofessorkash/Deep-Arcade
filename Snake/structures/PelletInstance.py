"""
TITLE:          PelletInstance.py (Snake)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Support Python file holding the pellet object instance for 
                the deep learning reinforcement simulation of Snake.
"""

from random import randint
import pygame

class PelletInstance(object):
    """ Object storing pellet instance to serve as player food. """
    def __init__(self):
        self.dim_x, self.dim_y = 240, 200
        self.sprite = pygame.image.load("structures/img/pellet.png")

    def get_pellet_position(self, game_session, player_instance):
        """ Method to grab current pellet position and check if pellet has been consumed by player. """
        rand_x, rand_y = randint(20, game_session.play_width - 40), randint(20, game_session.play_height - 40)
        self.dim_x, self.dim_y = rand_x - (rand_x % 20), rand_y - (rand_y % 20)
        if [self.dim_x, self.dim_y] not in player_instance.position:
            return self.dim_x, self.dim_y
        else:
            self.get_pellet_position(game_session, player_instance)

    def display_pellet(self, x_pos, y_pos, game_session):
        """ Method to display pellet sprite and position in game session. """
        game_session.play_display.blit(self.sprite, (x_pos, y_pos))
        pygame.display.update()