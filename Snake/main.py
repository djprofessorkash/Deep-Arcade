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

def get_score(score, scoreboard):
    """ Function to grab current high score from game session. """
    if score >= scoreboard:
        return score
    else:
        return scoreboard

def initialize_game(player_instance, game_session, pellet_instance, game_agent):
    """ Function to initialize game session with appropriate player and food parameters. """
    state_init_1 = game_agent.get_game_state(game_session, player_instance, pellet_instance)
    action_init = [1, 0, 0]
    player_instance.move_player(action_init, player_instance.dim_x, player_instance.dim_y, game_session, pellet_instance, game_agent)
    state_init_2 = game_agent.get_game_state(game_session, player_instance, pellet_instance)
    current_reward = game_agent.get_game_reward(player_instance, game_session.has_crashed)
    game_agent.save_state_to_memory(state_init_1, action_init, current_reward, state_init_2, game_session.has_crashed)
    game_agent.replay_from_memory(game_agent.memory)

def render_user_interface(game_session, score, scoreboard):
    """ Function to render the game's UI. """
    font = pygame.font.SysFont("Segoe UI", 20)
    font_bold = pygame.font.SysFont("Segoe UI", 20, True)
    score_text = font.render("SCORE: ", True, (0, 0, 0))
    score_text_number = font.render(str(score), True, (0, 0, 0))
    highest_score_text = font.render("HIGHEST SCORE: ", True, (0, 0, 0))
    highest_score_text_number = font_bold.render(str(scoreboard), True, (0, 0, 0))
    game_session.play_display.blit(score_text, (45, 440))
    game_session.play_display.blit(score_text_number, (120, 440))
    game_session.play_display.blit(highest_score_text, (190, 440))
    game_session.play_display.blit(highest_score_text_number, (350, 440))
    game_session.play_display.blit(game_session.background, (10, 10))

def render_game(player_instance, pellet_instance, game_session, scoreboard):
    """ Function to render the game board with complete mechanics. """
    pass

def _plot_game_results(counter_plot, score_plot):
    """ Helper function utilizing Seaborn to statistically plot game session results. """
    pass

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