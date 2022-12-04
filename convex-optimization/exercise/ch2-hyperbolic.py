import os
import argparse
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def gen2d(N=1000, range=[-1,1]):
    points = np.random.random((N, 2)) * (range[1] - range[0]) + (range[0],range[0])
    return points
    
def main():
    N = 10000
    points = gen2d(N, [-5,5])

    plt.scatter(points[:,0], points[:,1], s=2, alpha=0.3)
    prd = points[:,0] * points[:,1] 
    mask = prd >= 1
    plt.scatter(points[mask,0], points[mask,1], s=2, alpha=0.3, c='r')
    plt.show()
    print('done.')

if __name__ == '__main__':
    main()