# testing for q1

# dependencies
import pytest
import q1funcs
from numpy import random
import numpy as np
from scipy.linalg import toeplitz

# test 'tridiagLU' works
@pytest.mark.parametrize('m', [23, 154, 77])
def test_tridiagLU(m):
    random.seed(1878*m)
    c = 0
    d = 0
    while (c==0): # ensure values aren't zero
        c = random.uniform(-10*m,10*m)
    while (d==0):
        d = random.uniform(-10*m,10*m)
    a1 = np.zeros(m)
    a2 = np.zeros(m)
    a1[0] = c
    a1[1] = d
    a2[1] = d
    A0 = toeplitz(a1, a2) # construct tridiag A0
    U0 = q1funcs.tridiagLU(A0)
    U = np.triu(U0, k=0) # U from LU decomp.
    L = np.identity(m, dtype=A0.dtype) + np.tril(U0, k=-1) # L from LU decomp
    A = L @ U
    assert(np.allclose(L, np.tril(L))) #check L is lower triangular
    assert(np.allclose(U, np.triu(U))) #check U is upper triangular
    assert(np.linalg.norm(A - A0) < 1.0e-6) #check LU = A

#test 'tridiagLUsolve' works
@pytest.mark.parametrize('m', [33, 174, 83])
def test_tridiagLUsolve(m):
    random.seed(3453*m)
    c = 0
    d = 0
    while (c==0): #ensure values aren't zero
        c = random.uniform(-10*m,10*m)
    while (d==0):
        d = random.uniform(-10*m,10*m)
    a1 = np.zeros(m)
    a2 = np.zeros(m)
    a1[0] = c
    a1[1] = d
    a2[1] = d
    A = toeplitz(a1, a2)
    x0 = random.uniform(-10*m,10*m,m)
    b = A @ x0
    x = q1funcs.tridiagLUsolve(A, b) # solve the system
    assert(np.linalg.norm(A@x - b) < 1.0e-4) # check the solution is valid