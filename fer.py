''' Fer2013 Dataset class'''

from __future__ import print_function

import torch
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import pandas as pd


class FER2013(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        if self.split == 'Training':
            self.data = pd.read_csv("./data/train-20240123-14902.csv")
            self.train_data = self.data['pixels']
            self.train_labels = self.data['Valence']

            # convert train_data to images.
            self.train_data = np.asarray(self.train_data)
            self.train_data = [list(map(int, string.split())) for string in self.train_data]
            # # Convert the list of lists to a list of NumPy arrays
            self.train_data = [np.array(inner_list) for inner_list in self.train_data]
            # Convert the list of 1-D arrays to a list of 2-D arrays (2x2)
            self.train_data = [arr.reshape(48, 48) for arr in self.train_data]
            # Adding a third dimension to each array
            self.train_data = [arr[:, :, np.newaxis] for arr in self.train_data]
            temp = list()
            for arr in self.train_data:
                arr = np.concatenate((arr, arr, arr), axis=2).astype('uint8')
                arr = Image.fromarray(arr)
                if self.transform is not None:
                    arr = self.transform(arr)  # transform train_data, for normalization.
                temp.append(arr)
            self.train_data = temp
            # Convert target to a float (assuming it's already a numerical value)
            self.train_labels = torch.tensor(self.train_labels, dtype=torch.float32)
        elif self.split == 'PublicTest':
            self.data = pd.read_csv("./data/publictest-20240122.csv")
            self.PublicTest_data = self.data['pixels']
            self.PublicTest_labels = self.data['Valence']
            self.PublicTest_data = np.asarray(self.PublicTest_data)

            # convert PublicTest_data to images.
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.PublicTest_data = [list(map(int, string.split())) for string in self.PublicTest_data]
            # # Convert the list of lists to a list of NumPy arrays
            self.PublicTest_data = [np.array(inner_list) for inner_list in self.PublicTest_data]
            # Convert the list of 1-D arrays to a list of 2-D arrays (2x2)
            self.PublicTest_data = [arr.reshape(48, 48) for arr in self.PublicTest_data]
            # Adding a third dimension to each array
            self.PublicTest_data = [arr[:, :, np.newaxis] for arr in self.PublicTest_data]
            temp = list()
            for arr in self.PublicTest_data:
                arr = np.concatenate((arr, arr, arr), axis=2).astype('uint8')
                arr = Image.fromarray(arr)
                if self.transform is not None:
                    arr = self.transform(arr)  # transform , for normalization.
                temp.append(arr)
            self.PublicTest_data = temp
            # Convert target to a float (assuming it's already a numerical value)
            self.PublicTest_labels = torch.tensor(self.PublicTest_labels, dtype=torch.float32)
        else:
            self.data = pd.read_csv("./data/privatetest-20240122.csv")
            self.PrivateTest_data = self.data['pixels']
            self.PrivateTest_labels = self.data['Valence']

            # convert PrivateTest_data to images.
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = [list(map(int, string.split())) for string in self.PrivateTest_data]
            # # Convert the list of lists to a list of NumPy arrays
            self.PrivateTest_data = [np.array(inner_list) for inner_list in self.PrivateTest_data]
            # Convert the list of 1-D arrays to a list of 2-D arrays (2x2)
            self.PrivateTest_data = [arr.reshape(48, 48) for arr in self.PrivateTest_data]
            # Adding a third dimension to each array
            self.PrivateTest_data = [arr[:, :, np.newaxis] for arr in self.PrivateTest_data]
            temp = list()
            for arr in self.PrivateTest_data:
                arr = np.concatenate((arr, arr, arr), axis=2).astype('uint8')
                arr = Image.fromarray(arr)
                if self.transform is not None:
                    arr = self.transform(arr)  # transform , for normalization.
                temp.append(arr)
            self.PrivateTest_data = temp

            # Convert target to a float (assuming it's already a numerical value)
            self.PrivateTest_labels = torch.tensor(self.PrivateTest_labels, dtype=torch.float32)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]
        # to return a PIL Image
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)
