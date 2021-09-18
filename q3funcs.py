import numpy as np
import cla_utils
import  matplotlib.pyplot as plt
from sklearn import datasets

# a sign function for simplicity 
def sgn(x):
    if x==0:
        return 1
    else:
        return np.sign(x)


def qr_factor_tri(A):
    # initialise values
    m,_ = A.shape
    V = np.zeros([2,m-1])
    R = 1.0*A

    # first m-2 loops using the 2x3 submatrices and householder reflectors
    for k in range(m-2):
        M = R[k:k+2,k:k+3]
        x = M[:,0]
        e1 = np.zeros(2)
        e1[0] = 1
        # construct v
        v = sgn(x[0])*np.linalg.norm(x)*e1 + x 
        # normalise v
        if np.linalg.norm(v) != 0.0:
            v /= np.linalg.norm(v)
        # apply reflectors
        M -= 2*np.outer(v, np.dot(v.T.conj(), M))
        # update values in R
        R[k:k+2,k:k+3] = M
        # store v_{k}
        V[:,k] = v

    # final 2x2 iteration
    M = R[m-2:,m-2:]
    x = M[:,0]
    e1 = np.zeros(2)
    e1[0] = 1
    v = sgn(x[0])*np.linalg.norm(x)*e1 + x
    if np.linalg.norm(v) != 0.0:
        v /= np.linalg.norm(v)
    M -= 2*np.outer(v, np.dot(v.T.conj(), M))
    R[m-2:,m-2:] = M
    V[:,m-2] = v
        
    return V, R

def qr_alg_tri(T, mod=False, shift=False):
    # initialise residual, m, rvec
    r = 1
    m, _ = T.shape
    rvec = []
    if shift==False:
        # iteration as defined in the question
        while r >= 1.0e-12:
            V, R = qr_factor_tri(T)
            # right mult by Q
            R[0:2,0:2] -= np.dot(2*R[0:2,0:2], np.outer(V[:,0], V[:,0]))
            for k in range(1,m-1):
                R[k-1:k+2,k:k+2] -= np.dot(2*R[k-1:k+2,k:k+2], np.outer(V[:,k], V[:,k]))
            
            T = R
            r = np.abs(T[m-1,m-2])
            rvec.append(r)

    if shift==True:
        I = np.eye(m)
        while r >= 1.0e-12:
            # shift values
            a = T[m-1,m-1]
            b = T[m-1,m-2]
            d = (T[m-2,m-2]-T[m-1,m-1])/2
            # implementing the shift
            mu = a - (sgn(d)*(b**2))/(np.abs(d) + np.sqrt(d**2 + b**2))
            T = T - mu*I

            V, R = qr_factor_tri(T)
            # right mult by Q
            R[0:2,0:2] -= np.dot(2*R[0:2,0:2], np.outer(V[:,0], V[:,0]))
            for k in range(m-1):
                R[k-1:k+2,k:k+2] -= np.dot(2*R[k-1:k+2,k:k+2], np.outer(V[:,k], V[:,k]))
            # shifted version
            T = R + (mu*I)
            r = np.abs(T[m-1,m-2])
            rvec.append(r)
    # modification referenced in part d
    if mod==False:
        return T
    if mod==True:
        return T, rvec

# the script in e, implemented as a function
# I did not return the diagonal matrix as it was not specified in the question to do so
def qr_alg(A, shift=False):
    m, _ = A.shape
    eigenvalues = np.zeros(m) # initialise eigenvalue vector
    Tvals = [] # stores T_{m,m-1}
    T = cla_utils.hessenberg(A) # reduce to tridiagonal 
    
    for k in range(m,1,-1):
        T0, r = qr_alg_tri(T, True, shift)
        eigenvalues[m-k] = T0[k-1,k-1] # obtain and store eigenvalue
        Tvals.append(r) # store Tm,m-1 values
        T = T0[:(k-1),:(k-1)] #reinitialise submatrix
    
    eigenvalues[m-1] = T[0,0]
 
    return Tvals, eigenvalues
 





