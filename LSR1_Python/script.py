import numpy as np
import lsr1tr as ls1

delta = 3
g = np.matrix(np.random.rand(1000,1))
g_old = np.matrix(10*np.random.rand(1000,1))
x_old = np.matrix(np.random.rand(1000,1))
x = np.matrix(np.random.rand(1000,1))
s = np.matrix(x-x_old)
y = np.matrix(g-g_old)
SY = np.transpose(s)*y
B = np.matrix(np.identity(1000))
YY = np.transpose(y)*y
SS = np.transpose(s)*s
Sc = s
Yc = y
ind = 3
lqn_nbupdate = 10
lqn_start = 1
lqn_end = 3
Hdiag =  1
itr =1
idebug = True
delta = 10
Bs = B*s
ls1.lsr1tr(g_old, g,x_old, x, SY, SS, YY, Sc, Yc, ind, lqn_nbupdate, lqn_start, lqn_end, Hdiag, itr, idebug, delta, Bs)