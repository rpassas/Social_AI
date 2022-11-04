import numpy as np
from util import *


def b_sigmoid_update(error, attention, prediction, behav_model):
    dif, avg_abs_error = error()
    attn_weighted_dif = attention @ dif
    updated_dif = behav_model @ attn_weighted_dif
    prediction = dynamic_sigmoid(prediction, updated_dif)
