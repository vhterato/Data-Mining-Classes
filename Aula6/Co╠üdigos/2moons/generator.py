import numpy as np 
import matplotlib.pyplot as plt

PI = 3.141592684

def genA(x):
    return [np.sin(x)+np.random.normal(0,0.3),x+np.random.normal(0,0.1)]
def genB(x):
    return [-np.sin(x)+np.random.normal(0,0.2),x+np.random.normal(0,0.1)+PI/2]

x_data = []
y_data = []
for i in range(5000):
    p = np.random.uniform(0,1)
    u = np.random.uniform(0,PI)
    if p>0.5:
        x_data.append(genA(u))
        y_data.append(0)
        plt.plot(x_data[-1][0],x_data[-1][1],marker="o",markerfacecolor="red",
                 markeredgecolor="black")
    else:
        x_data.append(genB(u))
        y_data.append(1)
        plt.plot(x_data[-1][0],x_data[-1][1],marker="o",markerfacecolor="blue",
                 markeredgecolor="black")

np.savetxt("xdata.dat",x_data)
np.savetxt("ydata.dat",y_data)
plt.axis("off")
plt.show()