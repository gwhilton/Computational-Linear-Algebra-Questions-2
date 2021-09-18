# dependencies
import numpy as np
import cla_utils
import scipy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from numpy import random
import matplotlib.pyplot as plt

# function used to investigate convergence
def laplacetest(m, conv=False):
    # case for residual norms
    if conv==False:
        # initialise values and construct our Graph matrix
        I = np.eye(m)
        G = np.random.randint(0,2,(m,m))
        L = scipy.sparse.csgraph.laplacian(G)
        A = I + L
        U = np.triu(A)
        # constructing a function f = \norm{I - M^{-1}A} to minimise within constraints
        # i.e. making I as similar to M^{-1}A as possible
        def f(x):
            norm = np.linalg.norm(I - np.linalg.inv(x*U)@A, 2)
            return norm

        nonlinear_constraint = NonlinearConstraint(f, 0, 1-1.0e-10)
        op = minimize(f, 0.6, method = 'trust-constr', constraints={nonlinear_constraint})
        mu = op.x
        # construct c referenced in equation (8)
        c = f(mu)
        # construct M as defined in the question
        M = mu*U
        # Compute M^{-1}A and its eigenvalues for analysis
        MinvA = np.linalg.inv(M) @ A
        evals = np.linalg.eig(MinvA)[0]

        # confirm that \abs{1- \lambda} < 1 for all eigenvalues \lambda
        for i in range(m):
            if np.abs(1-evals[i])>=1:
                print('Try a new c')
                break
        
        # define apply_pc case for this type of matrix, upper triangular        
        def apply_pc2(b):
            x = cla_utils.solve_U(M, b)
            return x

        x0 = random.randn(m)    
        b = A@x0

        # obtain results
        x1, n1, rnorms1, res1 = cla_utils.GMRES(A, b, 1000, 1.0e-3, True, True, apply_pc=apply_pc2)
        x2, n2, rnorms2, res2 = cla_utils.GMRES(A, b, 1000, 1.0e-3, True, True)

        # plotting norms on a log scale for comparison
        rlen1 = len(rnorms1)
        rlen2 = len(rnorms2)
        yc = [None]*rlen2
        for i in range(rlen2):
            yc[i] = c**(i+1)

        xx1 = range(1,rlen1+1)
        xx2 = range(1,rlen2+1)

        plt.plot(xx1, np.log(rnorms1), '--rx', label='Preconditioned')
        plt.plot(xx2, np.log(rnorms2), '--bx', label='Not Preconditioned')
        plt.plot(xx2, np.log(yc), '--gx', label='c^n')
        plt.xlabel('Iteration')
        plt.ylabel('Log(Residual Norm)')
        plt.title('Comparison of Residual Norms for GMRES and Preconditioned GMRES, m={}'.format(m))
        plt.legend()

        return plt
    
    # case for order of convergence
    else:
        # lists to store convergence orders of different sizes of matrices for preconditioned and not preconditioned
        q1 = []
        q2 = []
        # looping through the 10 random matrices, same method as above
        for i in range(1,11):
            t = m + 5*i
            I = np.eye(t)
            G = np.random.randint(0,2,(t,t))
            L = scipy.sparse.csgraph.laplacian(G)
            A = I + L
            U = np.triu(A)

            def f(x):
                norm = np.linalg.norm(I - np.linalg.inv(x*U)@A, 2)
                return norm

            nonlinear_constraint = NonlinearConstraint(f, 0, 1-1.0e-10)

            op = minimize(f, 0.6, method = 'trust-constr', constraints={nonlinear_constraint})
            mu = op.x
            c = f(mu)
            M = mu*U
            MinvA = np.linalg.inv(M) @ A
            evals = np.linalg.eig(MinvA)[0]

            for i in range(t):
                if np.abs(1-evals[i])>=1:
                    print('Try a new c')
                    break
        
            def apply_pc2(b):
                x = cla_utils.solve_U(M, b)
                return x

            x0 = random.randn(t)    
            b = A@x0

            x1, n1, rnorms1, res1 = cla_utils.GMRES(A, b, 1000, 1.0e-3, True, True, apply_pc=apply_pc2)
            x2, n2, rnorms2, res2 = cla_utils.GMRES(A, b, 1000, 1.0e-3, True, True)
            rlen1 = len(rnorms1)
            rlen2 = len(rnorms2)
            
            # estimating the order of convergence as detailed in my answer
            q1.append((np.log(np.abs((rnorms1[rlen1-1] - rnorms1[rlen1-2])/(rnorms1[rlen1-2]-rnorms1[rlen1-3]))))/(np.log(np.abs((rnorms1[rlen1-2] - rnorms1[rlen1-3])/(rnorms1[rlen1-3]-rnorms1[rlen1-4])))))
            q2.append((np.log(np.abs((rnorms2[rlen2-1] - rnorms2[rlen2-2])/(rnorms2[rlen2-2]-rnorms2[rlen2-3]))))/(np.log(np.abs((rnorms2[rlen2-2] - rnorms2[rlen2-3])/(rnorms2[rlen2-3]-rnorms2[rlen2-4])))))
            
        # plotting    
        xx = range(1,11)
        plt.plot(xx, q1, '--rx', label='Preconditioned')
        plt.plot(xx, q2, '--bx', label='Not Preconditioned')
        plt.xlabel('(Matrix Size-10)/5')
        plt.ylabel('Estimated Convergence Order')
        plt.title('Comparison of Convergence order for GMRES and Preconditioned GMRES')
        plt.legend()

        return plt

# examples given in the report
# laplacetest(10).show()
# laplacetest(35).show()
# laplacetest(100).show()
# laplacetest(10, True).show()

