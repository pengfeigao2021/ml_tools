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

def main():
    N = 10000
    points = np.random.random((N, 2)) * 2 - (1,1)
    plt.scatter(points[:,0], points[:,1], s=2, alpha=0.3)
    P = np.array([
        [1, 2],
        [2, 1],
    ])
    print('svd:', np.linalg.svd(P))
    prd = points @ P @ points.T
    prd = np.diag(prd)
    pdb.set_trace()
    mask = prd < 1
    plt.scatter(points[mask,0], points[mask,1], s=2, alpha=0.3, c='r')
    plt.title(np.linalg.inv(P))
    plt.show()
    print('done.')

if __name__ == '__main__':
    main()