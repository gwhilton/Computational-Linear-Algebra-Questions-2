# testing for q2
import pytest
import q2funcs
from numpy import random
import numpy as np
from scipy.linalg import toeplitz

# testing the solve function in 

@pytest.mark.parametrize('m', [35, 84, 143])
def test_q2solve(m):
    random.seed(778*m)
    c1 = 0 # ensures non-zero
    while (c1==0):
        c1 = random.uniform(-10*m,10*m)

    # construct A
    a1 = np.zeros(m)
    a2 = np.zeros(m)
    a1[0] = 1 + 2*c1
    a1[1] = -c1
    a2[1] = -c1
    A = toeplitz(a1, a2)
    A[0,m-1] = -c1
    A[m-1,0] = -c1
    x0 = random.uniform(-10*m,10*m,m)
    b = A @ x0
    x = q2funcs.q2solve(A, b) # solve the system
    assert(np.linalg.norm(A@x - b) < 1.0e-6) #check the solution is valid
