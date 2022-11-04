import numpy as np
import math


def matrix_sigmoid(x):
    '''
    Helper sigmoid function where the intercept is 0.5
    '''
    return 1 / (1 + np.exp(-x))


def chaotic_update(prob, threshold, error):
    if error > threshold:
        magnitude = np.random.choice([-1, 1]) * error
        r = round(np.random.uniform(0, magnitude), 3)
        p = [np.asarray([
            abs(i - r) if abs(i - r) <= 1 else i for i in prob])][0]
        return p
    return prob


def linear_update(prev, error):
    prob = error + prev
    for i in range(len(prob)):
        if prob[i] > 1:
            prob[i] = 1
        if prob[i] < 0:
            prob[i] = 0
    return prob


def sigmoid_update(center, error):
    center = dynamic_sigmoid(center, error)
    return center


def dynamic_sigmoid(i, x):
    '''
    Helper sigmoid function where the intercept is a value of i (list)
    '''
    y = np.exp(np.clip(-x, -100, 100))  # avoid runover into infinity.
    out = np.asarray([1 /
                      (1 + (
                          (1 - np.clip(i[j], 1e-50, 1-1e-50)) / np.clip(i[j], 1e-50, 1-1e-50)) * y[j])

                      for j in range(len(i))])  # changed clips to keep sigmoid function from getting stuck at 0 or 1
    return out


def entropy(prob):
    '''
    Calculate the entropy of a probability vector
    '''
    e = [-((p * math.log(p, 2)) + ((1-p) * math.log(1-p,  2)))
         for p in prob]
    attention = np.diag(e)
    return attention
