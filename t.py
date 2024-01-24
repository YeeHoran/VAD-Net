from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import h5pickle


with h5py.File('./data/CK_data.h5', 'w') as file:
    dataset = file.create_dataset('example_dataset', data=[1, 2, 3])

    # Use h5pickle to pickle the h5py object
    pickled_data = h5pickle.dumps(dataset)

# Example: Unpickle an h5py object
with h5py.File('', 'r') as file:
    # Use h5pickle to unpickle the h5py object
    self.data = h5pickle.loads(pickled_data)
# Now 'unpickled_dataset' is a valid h5py dataset