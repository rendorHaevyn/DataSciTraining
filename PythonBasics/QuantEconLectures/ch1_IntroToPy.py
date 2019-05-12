# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt

a = np.linspace(-np.pi,np.pi,100)
b = np.cos(a)
c = np.sin(a)

plt.plot(b,c)

# inner product
np.dot(a,b)
a @ b

# --- SECTION --- #
import pandas as pd
np.random.seed(1234)
d  = np.random.randn(5,2)
d  = np.random.randint(1,100,(5,2))
dates = pd.date_range(start='01/01/2019', periods=d.shape[0])
df = pd.DataFrame(d,columns=('Price','Weight'),index=dates)
df.mean()


# --- NUMBA - parallel and fastify code --- #

import numba
import time

def logreg(Y,X,w,iterations):
    for i in range(iterations):
        w -= np.dot(((1.0 /
                (1.0 + np.exp(-Y * np.dot(X, w)))
                -1.0) * Y), X)
    return w

@numba.jit(nopython=True, parallel=True)


@numba.jit(nopython=True, parallel=True)

@numba.njit('float64(float64,float64,int32,int32)', parallel=True)
def logregnumba(Y,X,w,iterations):
    for i in range(iterations):
        w -= np.dot(((1.0 /
                (1.0 + np.exp(-Y * np.dot(X, w)))
                -1.0) * Y), X)
    return w


X = np.random.randint(-100,100,1000)
Y = np.random.randn(1000)

X = np.random.ranf(1000)
Y = np.random.ranf(1000)

t1 = time.time()
w = logreg(Y,X,0,100000)
t2 = time.time()
f"LogReg Runtime: {t2-t1}"

t1 = time.time()
w = logregnumba(Y,X,0,100000)
t2 = time.time()
f"LogReg Runtime: {t2-t1}"

%timeit logreg(Y,X,0,100000)
%timeit logregnumba(Y,X,0.0,100000)
