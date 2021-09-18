# import dependencies
import q1funcs
import numpy as np

# function which solves the system Ax=b as we derived in the question
def q2solve(A, b):
    m, _ = A.shape

    # obtain c1 value
    c1 = -1*A[0,m-1]

    # construct u1, u2, v1, v2 and then T
    u1 = np.zeros(m, dtype=A.dtype)
    u2 = np.zeros(m, dtype=A.dtype)
    v1 = np.zeros(m, dtype=A.dtype)
    v2 = np.zeros(m, dtype=A.dtype)
    u1[m-1] = -c1
    u2[0] = -c1
    v1[0] = 1
    v2[m-1] = 1
    T = A - np.outer(u1,v1) - np.outer(u2, v2)

    # construct inv(T)b, inv(T)u1 and inv(T)u2
    invTb = q1funcs.tridiagLUsolve(T, b)
    invTu1 = q1funcs.tridiagLUsolve(T, u1)
    invTu2 = q1funcs.tridiagLUsolve(T, u2)

    #construct inv(M)b, inv(M)u_{2}
    invMb = invTb - (invTu1*np.inner(v1,invTb))/(1 + np.inner(v1, invTu1))
    invMu2 = invTu2 - (invTu1*np.inner(v1,invTu2))/(1 + np.inner(v1, invTu1))

    # compute the solution
    x = invMb - (invMu2* np.inner(v2, invMb))/(1 + np.inner(v2, invMu2))

    return x



    
    

        

    
    
    


    

    


