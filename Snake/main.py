"""
TITLE:          main.py (Snake)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Main Python file for controlling deep learning reinforcement simulation of Snake.
"""

import sys
import pygame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from random import randint
from keras.utils import to_categorical
from inspect import currentframe, getframeinfo
from structures import GameAgent, GameBoard, PelletInstance, PlayerInstance

display_option, speed = True, 50
pygame.font.init()

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
    game_session.play_display.fill((255, 255, 255))
    render_user_interface(game_session, game_session.score, scoreboard)
    player_instance.render_player(player_instance.position[-1][0], player_instance.position[-1][1], player_instance.food, game_session)
    pellet_instance.display_pellet(pellet_instance.dim_x, pellet_instance.dim_y, game_session)

def _plot_game_results(counter_plot, score_plot, save_name=None):
    """ Helper function utilizing Seaborn to statistically plot game session results. """
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([counter_plot])[0], np.array([score_plot])[0], color="b", x_jitter=0.1, line_kws={"color": "green"})
    ax.set(xlabel="games", ylabel="score")
    if save_name:
        plt.savefig("structures/distributions/{}.png".format(save_name), dpi=400)
    plt.show()

def main():
    """ Main run function. """
    pygame.init()
    game_agent = GameAgent.GameAgent()
    training_counter, scoreboard = 0, 0
    score_plot, counter_plot = list(), list()

    # NOTE: Editable params for model parameter tuning
    epochs_ = 150
    eps_ceil = 80
    rand_ceil = 200

    # for _training_round in range(2, 12):
    # if _training_round < 10:
    #     save_name = "dist00{}".format(_training_round)
    # else:
    #     save_name = "dist0{}".format(_training_round)
    while training_counter < epochs_:
        game = GameBoard.GameBoard(440, 440)
        player_1 = game.player
        food_1 = game.food

        initialize_game(player_1, game, food_1, game_agent)
        if display_option:
            render_game(player_1, food_1, game, scoreboard)

        while not game.has_crashed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            game_agent.epsilon = eps_ceil - training_counter
            old_state = game_agent.get_game_state(game, player_1, food_1)
            if randint(0, rand_ceil) < game_agent.epsilon:
                print("> Exploring at epoch {}.".format(training_counter))
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                print("> Exploiting at epoch {}.".format(training_counter))
                prediction = game_agent.model.predict(old_state.reshape((1, 11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            pygame.time.wait(speed)
            player_1.move_player(final_move, player_1.dim_x, player_1.dim_y, game, food_1, game_agent)
            new_state = game_agent.get_game_state(game, player_1, food_1)
            reward = game_agent.get_game_reward(player_1, game.has_crashed)
            game_agent.short_term_memory_trainer(old_state, final_move, reward, new_state, game.has_crashed)
            game_agent.save_state_to_memory(old_state, final_move, reward, new_state, game.has_crashed)
            scoreboard = get_score(game.score, scoreboard)

            if display_option:
                render_game(player_1, food_1, game, scoreboard)
                pygame.time.wait(speed)

        game_agent.replay_from_memory(game_agent.memory)
        training_counter += 1
        print("\nGAME: {}\tSCORE: {}\n".format(training_counter, game.score))
        score_plot.append(game.score)
        counter_plot.append(training_counter)
    # TODO: Save these as different weights files
    game_agent.model.save_weights("structures/data/custom_weights_EPO{}-EPS{}-RND{}.hdf5".format(epochs_, eps_ceil, rand_ceil))
    _plot_game_results(counter_plot, score_plot)

if __name__ == "__main__":
    main()
    print("Simulation complete. End run.")