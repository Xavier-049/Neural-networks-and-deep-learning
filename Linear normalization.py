import numpy as np

area = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                 106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
room = np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])

x1 = (area - area.min()) / (area.max() - area.min())
x2 = (room - room.min()) / (room.max() - room.min())

