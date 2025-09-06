import numpy as np
import matplotlib.pyplot as plt

H = np.load("homography.npy")
print(H)

plt.imshow(H, cmap="viridis")
plt.colorbar()
plt.show()
