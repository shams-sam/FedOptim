# https://fabrizioguerrieri.com/blog/surface-graphs-with-irregular-dataset/

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D


points = 500
data = np.zeros([points, 3])
x = np.random.rand(points)*100
y = np.random.rand(points)*100
z = np.sinc((x-20)/100*3.14) + np.sinc((y-50)/100*3.14)
triang = mtri.Triangulation(x, y)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.plot_trisurf(triang, z, cmap='jet')
ax.scatter(x, y, z, marker='.', s=10, c="black", alpha=0.5)
ax.view_init(elev=60, azim=-45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('../ckpts/non-convex.png', dpi=300)
