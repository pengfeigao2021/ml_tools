import os
import argparse
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re
import pdb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def generate_line(a=2, b = 3, sigma=2):
    X = np.linspace(0, 10, 10)
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    bias = np.random.normal(loc=0, scale=sigma, size=X.shape).astype(np.int32)
    X += bias
    X = np.sort(X)

    Y = X * a + b + noise
    Y_gt = X * a + b
    plt.plot(X, Y, label='y_observe')
    plt.plot(X, Y_gt, alpha=0.3, color='r', label='y_gt')
    return X, Y
    

def main():
    X, Y = generate_line()

    p = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    n = X.shape[0]
    p.fit(X.reshape(n, 1), Y)

    x_pred = np.linspace(0, 10, 200)
    y_pred, y_std = p.predict(x_pred.reshape(200, 1), return_std=True)
    plt.plot(x_pred, y_pred, alpha=0.6, color='y', label='y_pred')
    plt.plot(x_pred, y_pred + y_std, '--', alpha=0.6, color='y', label='y+std')
    plt.plot(x_pred, y_pred - y_std, '--', alpha=0.6, color='y', label='y-std')
    plt.fill_between(
        x_pred,
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    print(p)
    plt.legend()
    plt.show()

    print('done.')

if __name__ == '__main__':
    main()