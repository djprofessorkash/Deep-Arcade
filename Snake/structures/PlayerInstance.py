"""
TITLE:          PlayerInstance.py (Snake)
AUTHOR:         Aakash Sudhakar

SUMMARY:        Support Python file holding the player object instance for 
                the deep learning reinforcement simulation of Snake.
"""

from numpy import array_equal
import pygame

def player_eat_food(player_instance, pellet_instance, game_session):
    """ Function to facilitate food consumption and player growth across game session. """
    if player_instance.dim_x == pellet_instance.dim_x and player_instance.dim_y == pellet_instance.dim_y:
        pellet_instance.get_pellet_position(game_session, player_instance)
        player_instance.has_eaten = True
        game_session.score += 1

class PlayerInstance(object):
    """ Object storing player instance to progress through the game session. """
    def __init__(self, game_session):
        dim_x, dim_y = 0.45 * game_session.play_width, 0.5 * game_session.play_height
        self.dim_x, self.dim_y = dim_x - (dim_x % 20), dim_y - (dim_y % 20)
        self.position = list()
        self.position.append([self.dim_x, self.dim_y])
        self.food = 1
        self.has_eaten = False
        self.image = pygame.image.load("structures/img/SnakeBody2.png")
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
        if array_equal(move_action, POTENTIAL_MOVES[0]):
            return self.delta_x, self.delta_y
        elif array_equal(move_action, POTENTIAL_MOVES[1]) and self.delta_y == 0:
            return [0, self.delta_x]
        elif array_equal(move_action, POTENTIAL_MOVES[1]) and self.delta_x == 0:
            return [-self.delta_y, 0]
        elif array_equal(move_action, POTENTIAL_MOVES[2]) and self.delta_y == 0:
            return [0, -self.delta_x]
        elif array_equal(move_action, POTENTIAL_MOVES[2]) and self.delta_x == 0:
            return [-self.delta_y, 0]
