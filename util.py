import numpy as np

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# Returns the indices of every occurrence of array b in array a
# Example:
# >>> a = np.array([0, 1, 2, 3, 0, 1, 2, 4])
# >>> b = np.array([0, 1, 2])
# >>> find_subarray(a, b)
# array([0, 4])
def find_subarray(a, b):
    temp = rolling_window(a, len(b))
    result = np.where(np.all(temp == b, axis=1))
    return result[0]