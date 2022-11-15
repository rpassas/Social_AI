import numpy as np
import math
import random
import scipy

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


def eigen_components(a=None):
    if not a:
        a = random.uniform(-0.99, 0.99)
    if a > 1 or a < -1:
        raise ValueError("a must be between -0.99 and 0.99")
    a2 = a**2
    b = math.sqrt(1-a2)
    # if you want a complex output
    #eig1 = complex(a, -b)
    #eig2 = complex(a, b)
    return a, b


def get_orbit_matrix(model_var=None, basis_mat=None):
    a, b = eigen_components(model_var)
    if basis_mat == None:
        basis_mat = np.random.rand(2, 2)
    if len(basis_mat) != 2 or len(basis_mat[0]) != 2:
        raise ValueError("basis matrix must have shape (2,2)")
    A = np.array([[a, -b],
              [b, a]])
    basis_mat_i = np.linalg.inv(basis_mat)
    M = basis_mat_i@A@basis_mat
    return M.real

def dynamic_transition(prior, M, error):
    prior_t = scipy.linalg.fractional_matrix_power(M, error).real@prior
    return prior_t
    