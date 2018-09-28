import numpy as np

if __name__ == '__main__':
    print(np.array([1, 2, 3]))
    array2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(array2d.shape)
    array2d.reshape(shape=(1, 6))
    print(array2d)
