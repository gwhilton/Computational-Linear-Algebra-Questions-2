# dependencies
import q2funcs
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from numpy import save

# function to implement the solution method generated in 2

def wavesol(N, M, delT, r, plot=True):
    # initialise u, w_n, f, x (for plotting)
    u = np.zeros(M)
    w_n = np.zeros(M)
    w_n1 = np.zeros(M)
    f = np.zeros(M)
    x = np.linspace(0, 1, M)
    num = np.floor(N/r)

    # initial condition functions
    def u_0(x):
        val = 0
        return val
    def u_1(x):
        val = np.cos(2*np.pi*x)
        return val

    # C1
    c1 = ((M*delT)**2)/4

    #A
    a1 = np.zeros(M)
    a2 = np.zeros(M)
    a1[0] = 1 + 2*c1
    a1[1] = -c1
    a2[1] = -c1
    A = toeplitz(a1, a2)
    A[0,M-1] = -c1
    A[M-1,0] = -c1
    
    # create our guesses
    for i in range(M):
        u[i] = u_0((i+1)/M)
        w_n[i] = u_1((i+1)/M)

    # initial f
    f[0] = w_n[0] + (M**2)*delT*(u[1] - 2*u[0] + u[M-1]) + c1*(w_n[1] - 2*w_n[0] + w_n[M-1])
    for i in range(1,M-1):
        f[i] = w_n[i] + (M**2)*delT*(u[i+1] - 2*u[i] + u[i-1]) + c1*(w_n[i+1] - 2*w_n[i] + w_n[i+1])
    f[M-1] = w_n[M-1] + (M**2)*delT*(u[0] - 2*u[M-1] + u[M-2]) + c1*(w_n[0] - 2*w_n[M-1] + w_n[M-2])

    #first sol
    
    w_n1 = q2funcs.q2solve(A, f)
    u += (delT/2)*(w_n + w_n1)
    count = 1

    # case for plotting

    if (plot==True):
        if count%num == 0:
            plt.plot(x, u)
            plt.show()
            count = 0

        for n in range(N-1): # iterative process
            w_n = w_n1

            f[0] = w_n[0] + (M**2)*delT*(u[1] - 2*u[0] + u[M-1]) + c1*(w_n[1] - 2*w_n[0] + w_n[M-1])
            for i in range(1,M-1):
                f[i] = w_n[i] + (M**2)*delT*(u[i+1] - 2*u[i] + u[i-1]) + c1*(w_n[i+1] - 2*w_n[i] + w_n[i+1])
            f[M-1] = w_n[M-1] + (M**2)*delT*(u[0] - 2*u[M-1] + u[M-2]) + c1*(w_n[0] - 2*w_n[M-1] + w_n[M-2])

            w_n1 = q2funcs.q2solve(A, f)
            u += (delT/2)*(w_n + w_n1)

            count += 1 
            if count%num == 0:
                plt.plot(x, u)
                plt.xlabel('x')
                plt.ylabel('u')
                plt.title('Wave Equation')
                plt.show()
                count = 0

    # case for saving            

    else:
        if count%num == 0:
            save('u1.npy', u)        
            count = 0

        for n in range(N-1): #iterative process
            w_n = w_n1

            f[0] = w_n[0] + (M**2)*delT*(u[1] - 2*u[0] + u[M-1]) + c1*(w_n[1] - 2*w_n[0] + w_n[M-1])
            for i in range(1,M-1):
                f[i] = w_n[i] + (M**2)*delT*(u[i+1] - 2*u[i] + u[i-1]) + c1*(w_n[i+1] - 2*w_n[i] + w_n[i+1])
            f[M-1] = w_n[M-1] + (M**2)*delT*(u[0] - 2*u[M-1] + u[M-2]) + c1*(w_n[0] - 2*w_n[M-1] + w_n[M-2])

            w_n1 = q2funcs.q2solve(A, f)
            u += (delT/2)*(w_n + w_n1)

            count += 1 
            if count%num == 0:
                save('u{}.npy'.format(n+2), u)
                count = 0

    return

# example where the solution should be a standing wave.
# want delta T to be approximately delta x
wavesol(50, 200, 1/200, 5, True)