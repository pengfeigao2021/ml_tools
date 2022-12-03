import os
import sys
import tqdm
import json
import copy
import random
import time
import numpy as np
import csv

import pygame
from pygame.locals import *

from util import connected_domain
from util import five_in_a_row

CURDIR = os.path.dirname(os.path.realpath(__file__))

class CheckerBoard(object):
    """ A class that draw a checker board in memory """
    def __init__(self, datestr, outdir=CURDIR+'/data'):
        self.log_path = os.path.join(outdir, datestr+'.log')
        self.history=[]
        
        self.length = 600
        self.width = 600

        pygame.init()
        self.screen = pygame.display.set_mode((self.length,self.width))
        pygame.display.set_caption("Five In a Row")

        self.draw_lines(15, 15)

    def draw_lines(self, row, col):
        color = 0,0,0
        width = 1
        board = 20

        white = 255,255,255
        orange = 255,100,80
        black = 0,0,0

        self.points = {}
        self.grid_witdh = self.width / row
        # record all points
        for r in range(row):
            pos_x_start = board
            pos_y = (self.width / row) * r + board
            for p in range(col):
                self.points[(pos_x_start+p*(self.length/col), pos_y)] = (r,p)

        mouse_down_x = mouse_down_y = 0
        self.mouse_down_real = []
        self.machine_down_real = []
        delay_draw = False
        rtn = 0
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    mouse_down_x, mouse_down_y = event.pos
                    pos_real = self.check_mouse_down(mouse_down_x, mouse_down_y)
                    if pos_real is not None:
                        self.mouse_down_real.append(pos_real)
                        self.history.append(self.points[tuple(pos_real)])

                        # when you take one step, machine will take one step too
                        pos_machine = self.take_one_step()
                        delay_draw = True
                        if pos_machine is not None:
                            self.machine_down_real.append(pos_machine)
                            self.history.append(self.points[tuple(pos_machine)])

            self.screen.fill((218,165,32))

            # draw 15*15 lines
            for r in range(row):
                pos_y = (self.width / row) * r + board
                pos_x_start = board
                pos_x_end = (self.length/col)*(col-1) + board
                pygame.draw.line(self.screen, color, (pos_x_start,pos_y), (pos_x_end,pos_y), width)
            for c in range(col):
                pos_x = (self.length / col) * c + board
                pos_y_start = board
                pos_y_end = (self.width/row)*(row-1) + board
                pygame.draw.line(self.screen, color, (pos_x,pos_y_start), (pos_x,pos_y_end), width)

            # draw five black points for positioning
            for pos in [(col/5,row/5), (col/5,row/5*4-1), (col/5*4-1,row/5),
                    (col/5*4-1,row/5*4-1), (col/2,row/2)]:
                for real_pos in self.points:
                    if pos == self.points[real_pos]:
                        real_pos = real_pos
                        color = 0,0,0
                        radius = 5
                        point_width = 0
                        pygame.draw.circle(self.screen, color, real_pos, radius, point_width)
                        break

            # draw points that you input and machine take
            for point in self.mouse_down_real:
                pygame.draw.circle(self.screen, black, point, self.grid_witdh/2-1, 0)

            for point in self.machine_down_real:
                if delay_draw and point == self.machine_down_real[-1]:
                    continue
                pygame.draw.circle(self.screen, white, point, self.grid_witdh/2-1, 0)

            if len(self.mouse_down_real) > 0:
                point = self.mouse_down_real[-1]
                pygame.draw.circle(self.screen, orange, point, self.grid_witdh/10, 0)
                point = self.machine_down_real[-1]
                pygame.draw.circle(self.screen, orange, point, self.grid_witdh/10, 0)

            pygame.display.update()

            # machine will play fast than what you can image,
            # but i let him sleep several seconds pretending to think
            if delay_draw:
                # check is game over
                # take the check to this because the code will get in
                # here when begin a new round
                rtn = self.check_gameover()
                self.do_if_gameover(rtn)

                time.sleep(random.uniform(0.5,1))
                pygame.draw.circle(self.screen, white, self.machine_down_real[-1], self.grid_witdh/2-1, 0)
                delay_draw = False
                pygame.display.update()

                # check is game over again
                # this is check for machine 
                rtn = self.check_gameover()
                self.do_if_gameover(rtn)



    def check_mouse_down(self, mouse_down_x, mouse_down_y):
        available_points = set(self.points.keys()) - set(self.mouse_down_real) -\
                        set(self.machine_down_real)
        min_distance = self.grid_witdh ** 2
        p = (0,0)
        for point in available_points:
            current_distance = (point[0]-mouse_down_x)**2 + (point[1]-mouse_down_y)**2
            if current_distance < min_distance:
                min_distance = current_distance
                p = point
                # print point
                # print mouse_down_x,mouse_down_y
        if min_distance != self.grid_witdh ** 2:
            return p
        else:
            return None

    def get_line_count(self, pos, direction, occupied):
        c = 1
        npos = pos + direction
        while tuple(npos.tolist()) in occupied:
            c += 1
            npos = npos + direction

        npos = pos - direction
        while tuple(npos.tolist()) in occupied:
            c += 1
            npos = npos - direction

        return c

    def get_greedy_score(self, pos, white_prototypes, black_prototypes):
        pos = np.array(pos)
        DIR = np.array([
            [0,1],
            [0,-1],
            [1,0],
            [-1,0],
            [-1,1],
            [1,1],
            [1,-1],
            [-1,-1],
        ])
        LINE_DIR = np.array([
            [0,1],
            [1,0],
            [-1,1],
            [1,1],
        ])
        score = 0
        for d in LINE_DIR:
            s = self.get_line_count(pos, d, white_prototypes)
            score += s

        # defend black
        for d in LINE_DIR:
            s = self.get_line_count(pos, d, black_prototypes)
            if s >= 4:
                score += 100
        return score

    def greedy_agent_step(self):
        """greedy algorithm.
        """
        available_points = set(self.points.keys()) - set(self.mouse_down_real) -\
                        set(self.machine_down_real)
        # score for white
        white_prototypes = set([self.points[k] for k in self.machine_down_real])
        black_prototypes = set([self.points[k] for k in self.mouse_down_real])
            
        max_score = None
        best_pos = None
        available_points = list(available_points)
        random.shuffle(available_points)
        for pos in available_points:
            score = self.get_greedy_score(self.points[pos], white_prototypes, black_prototypes)
            if max_score is None or score > max_score:
                max_score = score
                best_pos = pos

        # print('pos: {}, score: {}'.format(pos, max_score))
        return best_pos


    def take_one_step(self):
        available_points = set(self.points.keys()) - set(self.mouse_down_real) -\
                        set(self.machine_down_real)
        if len(available_points) != 0:
            # pick one point
            return self.greedy_agent_step()
        else:
            return None

    def check_gameover(self):
        # check is game over:
        # you or machine has five in a row
        if self.find_connected_domain(self.mouse_down_real):
            return 1

        if self.find_connected_domain(self.machine_down_real):
            return 2

        return 0

    def find_connected_domain(self, points):
        new_points = []
        for point in points:
            new_points.append(self.points[point])

        LINE_DIR = np.array([
            [0,1],
            [1,0],
            [-1,1],
            [1,1],
        ])

        for pos in new_points:
            for d in LINE_DIR:
                c = self.get_line_count(pos, d, new_points)
                if c >= 5:
                    return True
        return False

    def find_five_in_a_row(self, points):
        # First version
        # return True if there are five points
        """
        if len(points) < 5:
            return False
        else:
            return True
        """

        # Second version
        # find five point in a row
        if five_in_a_row(points) is None:
            return False
        else:
            return True


    def do_if_gameover(self, rtn):
        color = 255,255,255
        if rtn == 0:
            pass
        else:
            # output history
            with open(self.log_path, 'w') as fout:
                history = copy.deepcopy(self.history)
                history.append((-1, -1))
                wt = csv.writer(fout)
                print('output to {}'.format(self.log_path))
                wt.writerows(history)

            myfont = pygame.font.Font(None, 70)
            textImage = myfont.render("Game Over", True, color)
            self.screen.blit(textImage, (self.length/3, self.width/3))
            pygame.display.update()
            time.sleep(3)
            sys.exit()

class Piece(object):
    """ An instance of Piece is a point in checker board """
    def __init__(self, pos, is_black):
        self.pos = pos
        if is_black:
            self.color = 0,0,0
        else:
            self.color = 255,255,255
    def pre_draw(self):
        pass



if __name__ == "__main__":
    import datetime
    datestr = '{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    checker_board = CheckerBoard(datestr)

