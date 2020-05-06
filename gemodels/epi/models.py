# """
# All general Epidemology model Sits here
# """

import numpy as np
from collections import namedtuple

# Defining growth models with required interfaces

################################################
# SIR Model
################################################

def c_SIR(alpha, beta, rate, time):
    f = alpha / (1 + beta * np.exp(-rate * time))
    return f


def p_SIR(params, time):
    f = params.alpha / (1 + params.beta * np.exp(-params.rate * time))
    return f


all_func = {
    'SIR': {
        'df_model':3,
        'parameters':namedtuple('Parameters',['alpha','beta','rate']),
        "curve": c_SIR,
        "parametric": p_SIR
    }
}
