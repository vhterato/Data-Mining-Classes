import numpy as np
import pandas as pd

data = pd.read_table("wdbc.data",sep=",",header=None)
rows, cols = data.shape

X = np.empty((rows,cols-2))
y = []
for j in range(cols-2):
    for i in range(rows):
        X[i][j] = data[j+2][i]

for i in range(rows):
    if data[1][i]=="M":
        y.append(0)
    else:
        y.append(1)

np.savetxt("xdata.dat",X)
np.savetxt("ydata.dat",y)

print(X.shape)
print(len(y))