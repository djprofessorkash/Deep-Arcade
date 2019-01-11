"""
TITLE:          main.py (Snake)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Main Python file for controlling deep learning reinforcement simulation of Snake.
"""

import sys
import pygame
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas as pd
import numpy as np
from random import randint
from keras.utils import to_categorical
from inspect import currentframe, getframeinfo
from structures import GameAgent

display_option, speed = True, 50
pygame.font.init()

class GameBoard(object):
    """ Object structure storing the GameBoard session. """
    def __init__(self, play_width, play_height):
        pygame.display.set_caption("SnakeGen")
        self.play_width = play_width
        self.play_height = play_height
        self.play_display = pygame.display.set_mode((play_width, play_height + 60))
        self.background = pygame.image.load("structures/img/background.png")
        self.has_crashed = False
        self.player = PlayerInstance(self)
        self.food = PelletInstance()
        self.score = 0


class PlayerInstance(object):
    """ Object storing player instance to progress through the game session. """
    def __init__(self, game_session):
        dim_x, dim_y = 0.45 * game_session.play_width, 0.5 * game_session.play_height
        self.dim_x, self.dim_y = dim_x - (dim_x % 20), dim_y - (dim_y % 20)
        self.position = list()
        self.position.append([self.dim_x, self.dim_y])
        self.food = 1
        self.has_eaten = False
        self.image = pygame.image.load("structures/img/SnakeBody.png")
        self.delta_x, self.delta_y = 20, 0

    def update_relative_position(self, x_pos, y_pos):
        """ Method to update player's relative position across game session. """
        if self.position[-1][0] != x_pos or self.position[-1][1] != y_pos:
            if self.food > 1:
                for iterator in range(0, self.food - 1):
                    self.position[iterator][0], self.position[iterator][1] = self.position[iterator+1]
            self.position[-1][0], self.position[-1][1] = x_pos, y_pos
    
    def render_player(self, x_pos, y_pos, pellets, game_session):
        """ Method to continuously display player instance across game session. """
        self.position[-1][0], self.position[-1][1] = x_pos, y_pos
        if game_session.has_crashed == False:
            for iterator in range(pellets):
                x_pos_curr, y_pos_curr = self.position[len(self.position)-iterator-1]
                game_session.play_display.blit(self.image, (x_pos_curr, y_pos_curr))
            pygame.display.update()
        else:
            pygame.time.wait(300)

    def move_player(self, move_action, x_pos, y_pos, game_session, pellets, game_agent):
        """ Method that stores logic to perform player positional move. """
        move_vector = [self.delta_x, self.delta_y]
        if self.has_eaten:
            self.position.append([self.dim_x, self.dim_y])
            self.has_eaten = False
            self.food += 1
        move_vector = self._move_logic(move_vector, move_action)
        self.delta_x, self.delta_y = move_vector
        self.dim_x = x_pos + self.delta_x
        self.dim_y = y_pos + self.delta_y

        if self.dim_x < 20 or self.dim_x > game_session.play_width - 40 or self.dim_y < 20 or self.dim_y > game_session.play_height - 40 or [self.dim_x, self.dim_y] in self.position:
            game_session.has_crashed = True
        player_eat_food(self, pellets, game_session)
        self.update_relative_position(self.dim_x, self.dim_y)

    def _move_logic(self, move_vector, move_action):
        """ Helper method to run logical switch-cases that assess directional player movement. """
        POTENTIAL_MOVES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if np.array_equal(move_action, POTENTIAL_MOVES[0]):
            return self.delta_x, self.delta_y
        elif np.array_equal(move_action, POTENTIAL_MOVES[1]) and self.delta_y == 0:
            return [0, self.delta_x]
        elif np.array_equal(move_action, POTENTIAL_MOVES[1]) and self.delta_x == 0:
            return [-self.delta_y, 0]
        elif np.array_equal(move_action, POTENTIAL_MOVES[2]) and self.delta_y == 0:
            return [0, -self.delta_x]
        elif np.array_equal(move_action, POTENTIAL_MOVES[2]) and self.delta_x == 0:
            return [-self.delta_y, 0]


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
    game_session.play_display.fill((255, 255, 255))
    render_user_interface(game_session, game_session.score, scoreboard)
    player_instance.render_player(player_instance.position[-1][0], player_instance.position[-1][1], player_instance.food, game_session)
    pellet_instance.display_pellet(pellet_instance.dim_x, pellet_instance.dim_y, game_session)

def _plot_game_results(counter_plot, score_plot):
    """ Helper function utilizing Seaborn to statistically plot game session results. """
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([counter_plot])[0], np.array([score_plot])[0], color="b", x_jitter=0.1, line_kws={"color": "green"})
    ax.set(xlabel="games", ylabel="score")
    plt.show()

# def _update_screen():
#     """ Helper function to update physical screen. """
#     pygame.display.update()

def main():
    """ Main run function. """
    # print("First check: ", getframeinfo(currentframe()).lineno)
    pygame.init()
    game_agent = GameAgent.GameAgent()
    training_counter = 0
    score_plot, counter_plot = list(), list()
    scoreboard = 0

    # print("Second check: ", getframeinfo(currentframe()).lineno)
    while training_counter < 150:
        game = GameBoard(440, 440)
        player_1 = game.player
        food_1 = game.food
        # print("Third check: ", getframeinfo(currentframe()).lineno)

        initialize_game(player_1, game, food_1, game_agent)
        if display_option:
            render_game(player_1, food_1, game, scoreboard)
        # print("Fourth check: ", getframeinfo(currentframe()).lineno)

        while not game.has_crashed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                else:
                    game_agent.epsilon = 80 - training_counter
                    old_state = game_agent.get_game_state(game, player_1, food_1)
                    if randint(0, 200) < game_agent.epsilon:
                        final_move = to_categorical(randint(0, 2), num_classes=3)
                    else:
                        prediction = game_agent.model.predict(old_state.reshape((1, 11)))
                        final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

                    # print("Fifth check: ", getframeinfo(currentframe()).lineno)
                    pygame.time.wait(speed)
                    player_1.move_player(final_move, player_1.dim_x, player_1.dim_y, game, food_1, game_agent)
                    new_state = game_agent.get_game_state(game, player_1, food_1)
                    reward = game_agent.get_game_reward(player_1, game.has_crashed)
                    game_agent.short_term_memory_trainer(old_state, final_move, reward, new_state, game.has_crashed)
                    game_agent.save_state_to_memory(old_state, final_move, reward, new_state, game.has_crashed)
                    scoreboard = get_score(game.score, scoreboard)
                    # print("Sixth check: ", getframeinfo(currentframe()).lineno)
                    if display_option:
                        render_game(player_1, food_1, game, scoreboard)
                        pygame.time.wait(speed)
            # print("Seventh check: ", getframeinfo(currentframe()).lineno)

        game_agent.replay_from_memory(game_agent.memory)
        training_counter += 1
        # print("Eighth check: ", getframeinfo(currentframe()).lineno)
        print("Game:", training_counter, "\tScore:", game.score)
        score_plot.append(game.score)
        counter_plot.append(training_counter)
    game_agent.model.save_weights("data/custom_weights.hdf5")
    print("Out-of-while-loop check: ", getframeinfo(currentframe()).lineno)
    _plot_game_results(counter_plot, score_plot)


if __name__ == "__main__":
    main()
    print("Main-run done check: ", getframeinfo(currentframe()).lineno)