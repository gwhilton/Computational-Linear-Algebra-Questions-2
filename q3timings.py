# timings for q3

# dependencies
import q3funcs
import cla_utils
import time
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# function which produces time plots for qr_alg vs pure_QR
def timetest1():
    # initialise lists for timings
    ya = [None]*5
    yq = [None]*5
    for i in range(1,6):
        A = datasets.make_spd_matrix(5*i)
        t1 = time.time()
        x = q3funcs.qr_alg(A, False)
        t2 = time.time()
        y = cla_utils.pure_QR(A, 10000, 1.0e-12)
        t3 = time.time()
        ya[i-1] = np.log(t2-t1)
        yq[i-1]= np.log(t3-t2)

    # plotting
    xx = range(1,6)
    plt.plot(xx, ya, '--rx', label='Unshifted Q3e Alg')
    plt.plot(xx, yq, '--bx', label='pure_QR')
    plt.xlabel('(Matrix size)/5')
    plt.ylabel('Time to converge in log(s)')
    plt.title('Timing plots for Q3e')
    plt.legend()

    return plt

# same format as above but comparing unshifted and shifted versions of qr_alg
def timetest2():
    ys = [None]*5
    yu = [None]*5
    for i in range(1,6):
        A = datasets.make_spd_matrix(5*i)
        t1 = time.time()
        x = q3funcs.qr_alg(A, True)
        t2 = time.time()
        y = q3funcs.qr_alg(A, False)
        t3 = time.time()
        ys[i-1] = np.log(t2-t1)
        yu[i-1] = np.log(t3-t2)
    
    xx = range(1,6)
    plt.plot(xx, ys, '--rx', label='Shifted')
    plt.plot(xx, yu, '--bx', label='Unshifted')
    plt.xlabel('(Matrix size)/5')
    plt.ylabel('Time to converge in log(s)')
    plt.title('Convergence comparison for Wilkinson Shift')
    plt.legend()

    return plt


# produces plots in report
# timetest1().show()
# timetest2().show()