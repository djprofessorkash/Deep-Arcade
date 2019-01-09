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
from structures import DeepQNetworkAgent, GameSession, PlayerInstance, PelletInstance

display_option, speed = False, 0
pygame.font.init()

def _update_screen():
    """ Helper function to update physical screen. """
    pygame.display.update()

def main():
    """ Main run function. """
    print("\nHELLO WORLD\n")

if __name__ == "__main__":
    main()