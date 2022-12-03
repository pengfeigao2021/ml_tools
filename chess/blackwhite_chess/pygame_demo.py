import os
import argparse
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re
import pdb
import pygame
from pygame.locals import *

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    pygame.init()

    size = width, height = 320, 240
    speed = [2, 2]
    black = 0, 0, 0

    screen = pygame.display.set_mode(size)

    # ball = pygame.image.load("intro_ball.gif")
    ball = pygame.image.load("/Users/AlexG/Documents/GitHub/ml-tools/chess/blackwhite_chess/intro_ball.gif")
    ballrect = ball.get_rect()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        ballrect = ballrect.move(speed)
        if ballrect.left < 0 or ballrect.right > width:
            speed[0] = -speed[0]
        if ballrect.top < 0 or ballrect.bottom > height:
            speed[1] = -speed[1]

        screen.fill(black)
        screen.blit(ball, ballrect)
        pygame.display.flip()

if __name__ == '__main__':
    main()