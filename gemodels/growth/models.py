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

################################################
# Richard Model
################################################

def c_richard(alpha, beta, rate, variation, time):
    f = (1 + beta * np.exp(-rate * time)) ** (1 / variation)
    f = alpha / f
    return f


def p_richard(params, time):
    f = (1 + params.beta * np.exp(-params.rate * time)) ** (1 / params.variation)
    f = params.alpha / f
    return f


################################################
# Gompertz Model
################################################

def c_gompertz(alpha, beta, rate, time):
    f = alpha * np.exp(beta * np.exp(-rate * time))
    return f


def p_gompertz(params, time):
    f = params.alpha * np.exp(params.beta * np.exp(-params.rate * time))
    return f


################################################
# Chapman Richard Model
################################################

def c_chapman_richard(alpha, beta, rate, variation, time):
    f = alpha * ((1 - beta * np.exp(-rate * time))**(1 / (1 - variation)))
    return f


def p_chapman_richard(params, time):
    f = params.alpha * ((1 - params.beta * np.exp(-params.rate * time))**(1 / (1 - params.variation)))
    return f


################################################
# Bass Diffusion Model
################################################

def c_bass(mass, alpha, beta, time):
    flux = np.exp(-(alpha + beta) * time)
    f = mass * (1 - flux) / (1 + (beta / alpha) * flux)
    return f


def p_bass(params, time):
    flux = np.exp(-(params.alpha + params.beta) * time)
    f = params.mass * (1 - flux) / (1 + (params.beta / params.alpha) * flux)
    return f


################################################
# Weibull Model
################################################

def c_weibull(alpha, beta, rate, slope, time):
    f = alpha - beta * np.exp(-rate * time ** slope)
    return f


def p_weibull(params, time):
    f = params.alpha -  params.beta * np.exp(- params.rate * time **  params.slope)
    return f


################################################
# Accumulation
################################################

all_func = {
    'logistic': {
        'df_model':3,
        'parameters':namedtuple('Parameters',['alpha','beta','rate']),
        "curve": c_logistic,
        "parametric": p_logistic
    },
    'richard': {
        'df_model':4,
        'parameters':namedtuple('Parameters',['alpha','beta','rate','variation']),
        "curve": c_richard,
        "parametric": p_richard
    },
    'gompertz': {
        'df_model':3,
        'parameters':namedtuple('Parameters',['alpha','beta','rate']),
        "curve": c_gompertz,
        "parametric": p_gompertz
    },
    'chapmanrichard': {
        'df_model':4,
        'parameters':namedtuple('Parameters',['alpha','beta','rate','variation']),
        "curve": c_chapman_richard,
        "parametric": p_chapman_richard
    },
    'bass': {
        'df_model':3,
        'parameters':namedtuple('Parameters',['mass','alpha','beta']),
        "curve": c_bass,
        "parametric": p_bass
    },
    'weibull': {
        'df_model':4,
        'parameters':namedtuple('Parameters',['alpha','beta','rate','slope']),
        "curve": c_weibull,
        "parametric": p_weibull
    }
}
