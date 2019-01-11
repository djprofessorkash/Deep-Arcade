"""
TITLE:          main.py (Snake)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Main Python file for controlling deep learning reinforcement simulation of Snake.
"""

import pygame
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
from structures import GameAgent, GameBoard, PlayerInstance, PelletInstance

display_option, speed = False, 0
pygame.font.init()

def player_eat_food(player_instance, pellet_instance, game_session):
    """ Function to facilitate food consumption and player growth across game session. """
    if player_instance.dim_x == pellet_instance.dim_x and player_instance.dim_y == pellet_instance.dim_y:
        pellet_instance.get_pellet_position(game_session, player_instance)
        player_instance.has_eaten = True
        game_session.score += 1



def _update_screen():
    """ Helper function to update physical screen. """
    pygame.display.update()

def main():
    """ Main run function. """
    pygame.init()
    game_agent = GameAgent()
    training_counter = 0
    score_plot, counter_plot = list(), list()
    scoreboard = 0

if __name__ == "__main__":
    main()