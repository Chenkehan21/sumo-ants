import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-10, 10, 0.1)
y = np.arange(-10, 10, 0.1)
x, y = np.meshgrid(x, y)
fig = plt.figure()
ax3d = Axes3D(fig)

def f1(x, y):
    return (1 * x)**2 + (1 * y)**2

def f2(x, y):
    return -x**2 - y**2

ax3d.plot_surface(x, y, f1(x, y), cmap=plt.cm.rainbow)
plt.show()