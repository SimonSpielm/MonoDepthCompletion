import time
from functools import wraps

import numpy as np


def append_path(path_x1, path_y1, distances1, t1, path_x2, path_y2, distances2, t2):
    path_x1 = np.append(path_x1, path_x2[1:])
    path_y1 = np.append(path_y1, path_y2[1:])
    distances2 += distances1[-1]
    distances1 = np.append(distances1, distances2[1:])
    t2 += t1[-1]
    t1 = np.append(t1, t2[1:])
    return path_x1, path_y1, distances1, t1


def fill_arrays(path_x, path_y, distances, t, message_length=1000):
    samples = len(path_x)
    if samples > message_length:
        raise Exception(
            "Max amount of samples is 1000. Consider increasing the MAX_SAMPLES macro. Remember to change the msg def accordingly"
        )

    path_x = np.append(path_x, np.full((message_length - samples), path_x[-1]))
    path_y = np.append(path_y, np.full((message_length - samples), path_y[-1]))
    distances = np.append(distances, np.full((message_length - samples), distances[-1]))
    t = np.append(t, np.full((message_length - samples), t[-1]))
    return path_x, path_y, distances, t


def stopwatch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The function '{func.__name__}' has required {elapsed_time:.6f}s.")
        return result
    return wrapper


def print_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Called the function '{func.__name__}'")
        result = func(*args, **kwargs)
        return result
    return wrapper

def valid_parsed_frame(input):
    if isinstance(input, (np.ndarray, list)):
        return np.any(input)
    return bool(input)