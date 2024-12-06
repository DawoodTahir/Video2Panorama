import numpy as np
import matplotlib.pyplot as plt 
# array = np.arange(0,36,dtype=np.int64).reshape(4,9)
# print(array.shape)


# print(array)
# array_1 =np.where(np.any(array > 10, axis=1))[0]
# print(array_1)


img = 255 * np.ones((1000,1000,3), np.uint8)


plt.imshow(img)
plt.axis('off')  # Optional: Turn off axis ticks
plt.show()