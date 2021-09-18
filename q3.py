# dependencies
import numpy as np
import cla_utils
import  matplotlib.pyplot as plt
from sklearn import datasets
import q3funcs
from prettytable import PrettyTable

# function to produce error result and heatmap
def qr_alg_tri_Aij(A):
    # perform algo
    T = cla_utils.hessenberg(A)
    T = q3funcs.qr_alg_tri(T, False, False)
    evals = np.linalg.eig(A)[0]
    # generate error
    error = np.linalg.norm(np.diagonal(T)) - np.linalg.norm(evals)
    error_table = PrettyTable(['Error of eigenvalues calculation'])
    error_table.add_row([error])
    
    # plotting
    plt.imshow(T, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap of T')
    
    return error_table, plt

# plots of Tm,m-1 with no shift
def qr_alg_plots(A):
    Tvals, _ = q3funcs.qr_alg(A, False)

    Tvals = [item for sublist in Tvals for item in sublist]
    x = np.linspace(-1,1,len(Tvals))
    
    # plotting
    plt.plot(x, Tvals)
    plt.xlabel('Relative position of Tm,m-1')
    plt.ylabel('Absolute value of Tm,m-1 when convergence was reached')
    plt.title('Plot of results')

    return plt

# plots of Tm,m-1 with shift
def qr_alg_wilkinson_plots(A):
    #algo
    Tvals, _ = q3funcs.qr_alg(A, True)
    Tvals = [item for sublist in Tvals for item in sublist]
    x = np.linspace(-1,1,len(Tvals))

    # plotting
    plt.plot(x, Tvals)
    plt.xlabel('Relative position of Tm,m-1')
    plt.ylabel('Absolute value of Tm,m-1 when convergence was reached')
    plt.title('Plot of results')

    return plt

# create matrices to test on

A1 = np.zeros([5,5])
for i in range(5):
    for j in range(5):
        A1[i,j] = 1/(3+i+j)

A2 = datasets.make_spd_matrix(16)

D = np.diag(np.arange(15,0,-1))
O = np.ones([15,15])
A = D + O

# produces plots in the report

err, plotT = qr_alg_tri_Aij(A1)
print(err)
plotT.show()
plt.close()

qr_alg_plots(A1).show()
plt.close()
qr_alg_wilkinson_plots(A1).show()
plt.close()

qr_alg_plots(A2).show()
plt.close()
qr_alg_wilkinson_plots(A2).show()
plt.close()

qr_alg_plots(A).show()
plt.close()
qr_alg_wilkinson_plots(A).show()
plt.close()




    
    
    
    


