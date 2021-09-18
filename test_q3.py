# testing for q3

# dependencies
import pytest
import q3funcs
import cla_utils
from numpy import random
import numpy as np
from scipy.linalg import toeplitz
from sklearn import datasets

# test qr_factor_tri
@pytest.mark.parametrize('m', [31, 64, 101])
def test_qr_factor_tri(m):
    random.seed(555*m)
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
    V, R = q3funcs.qr_factor_tri(A)
    assert(np.allclose(R, np.triu(R)))  # check R is upper triangular
    assert(np.linalg.norm(np.dot(R.T, R) - np.dot(A.T, A)) < 1.0e-6)

# test qr_alg_tri with no shift
@pytest.mark.parametrize('m', [5, 12, 35])
def test_qr_alg_tri(m):
    random.seed(3453*m)
    A = datasets.make_spd_matrix(m)
    T = cla_utils.hessenberg(A)
    evals = np.linalg.eig(A)[0]
    T0 = q3funcs.qr_alg_tri(T, False, False)
    assert(np.linalg.norm(np.diagonal(T0)) - np.linalg.norm(evals) < 1.0e-06) # check evals are similar

# test qr_alg with no shift
@pytest.mark.parametrize('m', [5, 13, 26])
def test_qr_alg(m):
    random.seed(42*m)
    A = datasets.make_spd_matrix(m)
    Tvals, estevals = q3funcs.qr_alg(A, False)
    evals = np.linalg.eig(A)[0]
    assert(np.linalg.norm(estevals) - np.linalg.norm(evals) < 1.0e-06) # check evals are similar

# test qr_alg with a shift
@pytest.mark.parametrize('m', [6, 12, 21])
def test_qr_alg_shifted(m):
    random.seed(420*m)
    A = datasets.make_spd_matrix(m)
    Tvals, wilkevals = q3funcs.qr_alg(A, True)
    evals = np.linalg.eig(A)[0]
    assert(np.linalg.norm(wilkevals) - np.linalg.norm(evals) < 1.0e-06) # check evals are similar