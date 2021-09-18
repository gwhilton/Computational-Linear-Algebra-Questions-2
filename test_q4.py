# testing for q4

# dependencies
import pytest
import cla_utils
from numpy import random
import numpy as np
from sklearn import datasets

# test the modified GMRES with can be found in cla_utils
@pytest.mark.parametrize('m', [21, 17, 5])
def test_GMRES(m):
    random.seed(876*m)
    # well conditioned matrix
    A = datasets.make_spd_matrix(m)
    x0 = random.randn(m)
    b = A@x0
    M = np.diag(np.diag(A))
    m = A.shape[0]

    # define apply_pc arg for diagonal preconditioning
    def apply_pc1(b):
        x = np.zeros(m)
        for i in range(m):
            x[i] = b[i]/M[i,i]
        return x
    
    x, n = cla_utils.GMRES(A, b, 1000, 1.0e-3, apply_pc=apply_pc1)
    assert(np.linalg.norm(A@x - b) < 1.0e-3) # check GMRES produces the correct solution

