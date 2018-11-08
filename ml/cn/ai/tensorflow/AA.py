import numpy as np

if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [99, 10, 11, 12]])
    print(a[np.array([0,2])])