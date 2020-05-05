# """
# All general Growth Models reside here
# """

import numpy as np
from collections import namedtuple

# Defining growth models with required interfaces

################################################
# Logistic Model
################################################

def c_logistic(alpha, beta, rate, time):
    f = alpha / (1 + beta * np.exp(-rate * time))
    return f


def p_logistic(params, time):
    f = params.alpha / (1 + params.beta * np.exp(-params.rate * time))
    return f


all_func = {
    'logistic': {
        'df_model':3,
        'parameters':namedtuple('Parameters',['alpha','beta','rate']),
        "curve": c_logistic,
        "parametric": p_logistic
    }
}
