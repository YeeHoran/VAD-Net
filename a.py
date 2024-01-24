from PIL import Image
import numpy as np

# Example list of 2-D NumPy arrays (2x2)
list_of_2d_arrays = [np.array([[10, 20], [30, 40]]), np.array([[40, 50], [60, 70]]), np.array([[70, 80], [90, 100]])]

list_of_2d_arrays = [arr[:, :, np.newaxis] for arr in list_of_2d_arrays]
array_3d = np.repeat(list_of_2d_arrays, repeats=3, axis=2)

# Convert the list of 2-D arrays to a list of 2x2 images
#list_of_images = [Image.fromarray(arr.astype('uint8')) for arr in list_of_2d_arrays]

print(array_3d)

