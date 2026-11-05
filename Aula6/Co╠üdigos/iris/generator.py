import numpy as np 

X = np.loadtxt("iris.data",usecols=range(0,4),delimiter=",")
np.savetxt("xdata.dat",X)

y_str = np.loadtxt("iris.data", usecols=4, delimiter=",", dtype=str)
classes, y_int = np.unique(y_str, return_inverse=True)
np.savetxt("ydata.dat", y_int)
