import numpy as np
from scipy import ndimage, misc
import re
import matplotlib.image as mpimg
import os
import math 
import pandas as pd

path = os.path.abspath('..')
def load_CWRU():
    import scipy.io as scio
    data = scio.loadmat(path + "\\data\\CWRU_V2.mat")

    X = data['X1'].reshape(15000, 28, 28,1)#X1是DE端数据，X2是FE端#数据数据集大小（15000，28，28，1）

    Y = data['Y'].reshape(15000, )
    index = [i for i in range(len(X))]
    np.random.shuffle(index)
    x = X[index]
    y = Y[index]
    print("CWRU数据形状", X.shape)
    print("CWRU标签形状", Y.shape)
    return x, y

def load_GS():
    import scipy.io as scio
    data = scio.loadmat(path + "\\data\\GS_V2.mat")
    X = data['X1'].reshape(10000, 28, 28,1)
    Y = data['Y'].reshape(10000, )
    index = [i for i in range(len(X))]
    np.random.shuffle(index)
    x = X[index]
    y = Y[index]
    print("GS数据形状", X.shape)
    print("GS标签形状", Y.shape)
    return x, y



def load_data_conv(dataset, datapath):
    if dataset == 'CWRU':
        return load_CWRU()
    elif dataset == 'GS':
        return load_GS()

    else:
        raise ValueError('Not defined for loading %s' % dataset)

def load_data(dataset, datapath):
    x, y = load_data_conv(dataset, datapath)
    return x.reshape([x.shape[0], -1]), y

def generate_data_batch(x, y=None, batch_size=256):
    index_array = np.arange(x.shape[0])
    index = 0
    while True:
        idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
        if y is None:
            yield x[idx]
        else: 
            yield x[idx], y[idx]

def generate_transformed_batch(x, datagen, batch_size=256):
    if len(x.shape) > 2:  # image
        gen0 = datagen.flow(x, shuffle=False, batch_size=batch_size)
        while True:
            batch_x = next(gen0)
            yield batch_x
    else:
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen0 = datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=batch_size)
        while True:
            batch_x = next(gen0)
            batch_x = np.reshape(batch_x, [batch_x.shape[0], x.shape[-1]])
            yield batch_x      

def generate(x, datagen, batch_size=256):
    gen1 = generate_data_batch(x, batch_size=batch_size)
    if len(x.shape) > 2:  # image
        gen0 = datagen.flow(x, shuffle=False, batch_size=batch_size)
        while True:
            batch_x1 = next(gen0)
            batch_x2 = next(gen1)
            yield (batch_x1, batch_x2)
    else:
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen0 = datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=batch_size)
        while True:
            batch_x1 = next(gen0)
            batch_x1 = np.reshape(batch_x1, [batch_x1.shape[0], x.shape[-1]])
            batch_x2 = next(gen1)
            yield (batch_x1, batch_x2)


