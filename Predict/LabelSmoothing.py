import numpy as np


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.shape  # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    b = label_smoothing(a)
    print(b)