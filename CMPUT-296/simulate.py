"""
Assignmnet 1, Q4
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


def plot_gaussian(x, y, filename=None):
    """
    Plot the multivariate Gaussian

    If filename is not given, then the figure is not saved.
    """
    # Note: there was no need to make this into a separate function
    # however, it lets you see how to define functions within the
    # main file, and makes it easier to comment out plotting if
    # you want to experiment with many parameter changes without
    # generating many, many graphs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c="red", marker="s")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    minlim, maxlim = -3, 3
    ax.set_xlim(minlim, maxlim)
    ax.set_ylim(minlim, maxlim)
    if filename is not None:
        fig.savefig("scatter" + str(dim) + "_n" + str(numsamples) + ".png")
    plt.show()


if __name__ == '__main__':

    # default
    dim = 1
    numsamples = 10

    if len(sys.argv) > 1:
        dim = int(sys.argv[1])
        if dim > 3:
            print("Dimension must be 3 or less; capping at 3")
        if len(sys.argv) > 2:
            numsamples = int(sys.argv[2])
    print("Running with dim = " + str(dim),
          " and numsamples = " + str(numsamples))

    # Generate data from (Univariate) Gaussian
    if dim == 1:
        # mean and standard deviation in one dimension
        mu = 0
        sigma = 1.0
        x = np.random.normal(mu, sigma, numsamples)
        y = np.zeros(numsamples,)
    else:
        # mean and standard deviation in three dimension
        print("Dimension not supported")
        exit(0)

    # TODO: Print the current estimate of the sample mean
    print(np.mean(x))
    # TODO: Print the current estimate of the sample variance
    print(np.var(x))
    # Print all in 2d space
    plot_gaussian(x, y)
    #plot_gaussian(x,y,"scatter" + str(dim) + "_n" + str(numsamples) + ".png")
