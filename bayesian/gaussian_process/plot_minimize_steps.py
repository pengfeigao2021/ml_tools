'''
Most people who set up a GP regression or classification model end up using the Squared-Exp or Rational Quadratic kernels. 
    They are a quick-and-dirty solution that will probably work pretty well for interpolating smooth functions when N is a
    multiple of D, and when there are no 'kinks' in your function. If your function happens to have a discontinuity or is 
    discontinuous in its first few derivatives (for example, the abs() function), then either your lengthscale will end up 
    being extremely short, and your posterior mean will become zero almost everywhere, or your posterior mean will have 
    'ringing' effects. Even if there are no hard discontinuities, the lengthscale will usually end up being determined by 
    the smallest 'wiggle' in your function - so you might end up failing to extrapolate in smooth regions if there is even 
    a small non-smooth region in your data.
'''
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
from sklearn.gaussian_process.kernels import RBF, Sum, WhiteKernel
rbf_kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(0.2, 10))
c_kernel = WhiteKernel(0.01, noise_level_bounds=[0.009, 0.011])
kernel = Sum(c_kernel, rbf_kernel)


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def generate_line(a=2, b = 3, sigma=0.5):
    X = np.linspace(0, 10, 100)
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    bias = np.random.normal(loc=0, scale=sigma, size=X.shape).astype(np.int32)
    X += bias
    X = np.sort(X)

    Y_gt = np.sin(2.0*X) * a + b
    Y = Y_gt + noise
    return X, Y, Y_gt

def f(X, Y, x):
    # interpolation
    mi = np.argmin(np.abs(X - x))
    mx = X[mi]
    if abs(mx - x) < 1e-3:
        return Y[mi]
    
    x_range = []
    i_range = []
    if mx > x:
        if mi > 0:
            x_range = [X[mi-1], mx]
            i_range = [mi-1, mi]
        else:
            return Y[mi]
    else:
        if mi < X.shape[0]-1:
            x_range = [mx, X[mi+1]]
            i_range = [mi, mi+1]
        else:
            return Y[mi]
    print('X:', X)
    print('i, x range:', i_range, x_range)
    y0 = Y[i_range[0]]
    y1 = Y[i_range[1]]
    return y0 + (y1 - y0) * (x - x_range[0]) / (x_range[1] - x_range[0])
    
def minimize(X, Y, G=None):
    if G is None:
        p = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    else:
        p = G
    p.fit(X.reshape(X.shape[0], 1), Y)
    print('kernel: ', p.kernel_)
    title = 'kernel: {}'.format(p.kernel_)

    x_pred = np.linspace(0, 10, 200)
    y_pred, y_std = p.predict(x_pred.reshape(200, 1), return_std=True)
    y_min = y_pred - 1.96 * y_std,
    yi = np.argmin(y_min)
    xi = x_pred[yi]

    plt.plot(x_pred, y_pred, alpha=0.6, color='y', label='y_pred')
    plt.title(title)
    plt.plot(x_pred, y_pred + y_std, '--', alpha=0.6, color='y', label='y+std')
    plt.plot(x_pred, y_pred - y_std, '--', alpha=0.6, color='y', label='y-std')
    plt.fill_between(
        x_pred,
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    return xi

def run_step(X, Y, Y_gt, x_observe, y_observe, step=1):
    plt.clf()
    plt.plot(X, Y, label='y_observe')
    plt.plot(X, Y_gt, alpha=0.3, color='r', label='y_gt')

    xi = minimize(x_observe, y_observe)
    if xi in x_observe:
        plt.title('xi({}) in obs: {}'.format(xi, xi in x_observe))
        xi = np.random.randint(np.min(X), np.max(X))
    yi = f(X, Y, xi)

    plt.plot(xi, yi, marker='o', markersize=12)
    plt.plot(x_observe, y_observe, linestyle='', marker='o', markersize=5, markerfacecolor='y')
    plt.legend()
    plt.ylim([-1, 10])
    plt.savefig('/Users/AlexG/Downloads/tmp/step_{}.png'.format(step))
    return [xi, yi]

def main():
    X, Y, Y_gt = generate_line()
    step=10
    x_observe, y_observe = X[::step], Y[::step]
    for i in range(25):
        xi, yi = run_step(X, Y, Y_gt, x_observe, y_observe, step=i)
        x_observe = np.array(x_observe.tolist() + [xi])
        y_observe = np.array(y_observe.tolist() + [yi])

    print('done.')

if __name__ == '__main__':
    main()