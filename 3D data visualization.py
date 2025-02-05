import matplotlib.pyplot as plt


fig=plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
plt.show()

import numpy as np

x=np.random.uniform(10,40,30)
y=np.random.uniform(100,200,30)
z=np.random.uniform(10,20,30)

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.scatter(x, y, z, c='b', marker="*")
plt.show()

x=np.random.uniform(10,40,30)
y=np.random.uniform(100,200,30)
z=2*x+y

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')

ax3d.scatter(x, y, z, c='b', marker="*")

ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z=2X+Y')

plt.show()

x=np.random.uniform(10,40,300)
y=np.random.uniform(100,200,300)
z=2*x+y

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')

ax3d.scatter(x, y, z, c='b', marker="*")

ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z=2X+Y')

plt.show()