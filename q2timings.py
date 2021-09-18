# timings for q2

#dependencies
import q2funcs
import cla_utils
import time
from numpy import random
import numpy as np
from scipy.linalg import toeplitz

def q2_timings(m):
    # similar set up to testing file
    random.seed(778*m)
    c1 = 0
    while (c1==0):
        c1 = random.uniform(-m*10,m*10)
    a1 = np.zeros(m)
    a2 = np.zeros(m)
    a1[0] = 1 + 2*c1
    a1[1] = -c1
    a2[1] = -c1
    A = toeplitz(a1, a2)
    A[0,m-1] = -c1
    A[m-1,0] = -c1
    x0 = random.uniform(-m*10,m*10,m)
    b = A @ x0

    # timings
    t1 = time.time()
    x1 = q2funcs.q2solve(A, b)
    t2 = time.time()
    x2 = cla_utils.solve_LUP(A, b)
    t3 = time.time()
    return print('Time for q2solve={}, Time for solve_LUP={}'.format(t2-t1, t3-t2))

# two examples used in report
# q2_timings(25)
# q2_timings(100)

