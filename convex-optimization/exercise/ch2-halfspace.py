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
    points = gen2d(N)
    plt.scatter(points[:,0], points[:,1], s=2, alpha=0.3)
    a = (0.5,0.5)
    b = (-0.5,0.3)
    plt.plot(a[0:1], a[1:], marker='o', markersize=12, color='y')
    plt.plot(b[0:1], b[1:], marker='o', markersize=12, color='y')
    d1 = np.linalg.norm((points - a), axis=1)
    d2 = np.linalg.norm((points - b), axis=1)
    mask = d1 > d2
    plt.scatter(points[mask,0], points[mask,1], s=2, alpha=0.3, c='r')
    plt.title('a={}, b={}'.format(a, b))
    plt.show()
    print('done.')

if __name__ == '__main__':
    main()