import numpy as np

import scipy.io as scio

from DSAFN import gaussian_noise_layer

import warnings
warnings.filterwarnings("ignore")
import os

path = os.path.abspath('..')


def CWRU_V2():
    data = scio.loadmat(path + "\\data\\CWRU_V2.mat")
    print("dataset CWRU_V2")
    x1 = data['X2']
    x2 = data['X1']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x1 = gaussian_noise_layer(x1, 0.01)
    x2 = gaussian_noise_layer(x2, 0.01)
    Y = Y.reshape(15000, )
    # print(Y1)
    # print(Y1.shape)
    Y = Y[index]
    Y = Y.reshape(1, 15000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2], Y
    # return [x1], Y

def CWRU_V3():
    data = scio.loadmat(path + "\\data\\CWRU_V3.mat")
    print("dataset CWRU_V3")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x3 = x3[index]
    x1 = gaussian_noise_layer(x1, 0.01)
    x2 = gaussian_noise_layer(x2, 0.01)
    x3 = gaussian_noise_layer(x3, 0.01)
    Y = Y.reshape(15000, )
    Y = Y[index]
    Y = Y.reshape(1, 15000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2,x3], Y


def load_data_conv(dataset):
    print("load:", dataset)#load: MNIST_USPS_COMIC
    if dataset == 'CWRU_V3':
        return CWRU_V3()
    elif dataset == 'CWRU_V2':
        return CWRU_V2()
    else:
        raise ValueError('Not defined for loading %s' % dataset)
