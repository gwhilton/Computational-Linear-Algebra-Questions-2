# import dependencies
import numpy as np

# A function which returns U, containing U values on and above the diagonal
# and the L values strictly below the diagonal. These values form the LU decomposition
# of the input A

# Used this inplace scheme to maximise efficiency and reduce the number of zeros stored

def tridiagLU(A):
    m, _ = A.shape
    # initialise U, c, d
    c = A[0,0]
    d = A[0,1]
    U = np.zeros([m,m], dtype=A.dtype)

    # initialise first element of U
    U[0,0] = c
    for k in range(1,m):
        U[k,k-1] = d/U[k-1,k-1] # adding L values
        U[k,k] = c - U[k,k-1]*d # adding diagonal U values
        U[k-1,k] = d # adding U values above the diagonal

    return U

#A function which solves Ax=b when A is a symmetric tridiagonal matrix. 
# It uses as few zeros as possible and as few computations as possible

def tridiagLUsolve(A, b):
    m, _ = A.shape
    # initialise c, d, L, U, y, x
    c = A[0,0]
    d = A[0,1]
    U = np.zeros(m, dtype=A.dtype)
    y = np.zeros(m, dtype=A.dtype)
    x = np.zeros(m, dtype=A.dtype)
    l = 0
    # initialise first element of U and
    U[0] = c
    y[0] = b[0]
    for k in range(1,m):
        # LU decomposition
        l = d/U[k-1] 
        U[k] = c - d*l    
        # Forward substitution to solve Ly=b
        y[k] = b[k] - l*y[k-1]

    # initialise first value for back substitution
    x[m-1] = y[m-1]/U[m-1]
    # Back substitution to solve Ux=y
    for k in range(m-2,-1,-1):
        x[k] = (y[k] - d*x[k+1])/U[k]

    return x


