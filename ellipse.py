import numpy as np
import scipy as sp
from scipy import linalg

# kpts: [u, v, a, b, c] where a(x-u)(x-u)+2b(x-u)(y-v)+c(y-v)(y-v)=1
# example

def kpToEllipse(kp):
    x = kp[0]
    y = kp[1]
    a = kp[2]
    b = kp[3]
    c = kp[4]
    matA = np.array([[a,b/2],[b/2,c]]) # creates matrix A
    w,v = np.linalg.eig(matA) # gets eigenvalues and vectors
    v= -1*v
    det = np.linalg.det(v)
    
    theta = v[1][0]/v[0][0]
    degs = theta*180/np.pi

    xax = (1/w[0])**.5
    yax = (1/w[1])**.5

    return degs, xax, yax

degs, xax, yax = kpToEllipse(kptEx)



    
