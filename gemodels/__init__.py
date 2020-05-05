from pkg_resources import get_distribution, DistributionNotFound
import logging

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import f as statf


# defining stat function to avoid dependency on any other package

class StatError(Exception):
    """Base Stat Error"""
    pass


class ModelError(Exception):
    """Base Model Error"""
    pass


def check_data(data):
    if isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            y = data.reshape(1, -1)
            t = np.arange(1, len(y))
        else:
            # assuming first array is time and second is values
            # todo: validate this later
            y = data[:, 1]
            t = data[:, 0]
    elif isinstance(data, list):
        y = np.array(data)
        t = np.arange(1, len(y))
    else:
        raise ModelError('Cannot Parse Data : \n', data)
    return y, t

def check_data_1d(x):
    if isinstance(x, list):
        x = np.array(x)
    elif isinstance(x, np.ndarray):
        if len(x.shape) > 1:
            raise StatError('Data must be 1D only.')
    return x


class ModelStats(object):

    def __init__(self, name=None,p_alpha=None,
                 tolerance=1e-4, keep_stat=True, digits=2):

        # Class variables
        # self.confidence_measure = confidence_measure
        self.identifier = name
        self.confidence_alpha = p_alpha if p_alpha is not None else 0.05
        self.tolerance = tolerance
        self.keep_stat = keep_stat
        self.digits = digits

        # Deviation Measures
        self.me = 0  # Mean Error
        self.bias = 0  # Multiplicative bias
        self.mae = 0  # Mean Absolute Error
        self.mad = 0  # Mean Absolute Deviance
        self.mape = 0  # Mean Average Percentage Error
        self.rmse = 0  # Root Mean Square Error
        # may implement median versions and tweedie deviance,
        # correlation coeffcient, log errors, skill score, LESP
        # refer : https://www.cawcr.gov.au/projects/verification/#Methods_for_dichotomous_forecasts
        #

        # Model Measures
        self.r2_val = 0  # R2
        self.adjr2_val = 0  # Adj R2
        self.aic = 0  # AIC
        self.bic = 0  # BIC
        self.fstat = (0, 0)  # F-statistics
        self.ndf = 0  # Degree of freedom
        self.mdf = 0  # Model Degree of Freedom
        self.loglik = 0  # Log Likelihood # May generalize this

    def _deviation_measures(self, y_act, y_fit):
        error = y_act - y_fit
        n = len(y_act)
        self.me = np.mean(error)
        self.bias = np.mean(y_fit) / np.mean(y_act)
        self.mae = np.mean(np.abs(error)) / n
        self.mad = np.mean(np.abs(error - self.me))
        self.mape = np.mean(np.abs(error / y_act)) * 100
        self.rmse = np.sqrt(np.mean(error ** 2))
        self.r2_val = 1.0 - (np.sum(error ** 2) / ((n - 1.0) * np.var(y_act, ddof=1)))

    def _model_measures(self, y_act, y_fit, ndf, mdf):
        error = y_act - y_fit
        n = len(y_act)
        fdist = statf(n - 1, n - 1)
        alpha = self.confidence_alpha

        # degree of freedom
        self.ndf = ndf
        self.mdf = mdf

        # Adjusted R squared
        self.adjr2_val = 1 - (np.var(error, ddof=1) * (ndf - 1)) / (np.var(y_act, ddof=1) * (ndf - mdf - 1))

        # F statistics
        f_val = np.var(y_fit, ddof=1) / np.var(y_act, ddof=1)
        f_p_value = 2 * min(fdist.cdf(alpha), 1 - fdist.cdf(alpha))
        self.fstat = (f_val, f_p_value)

        # Likely to add Levene's , Barttlets, Brown–Forsythe variants  and Box - M test
        # self.loglik = None
        # self.aic = None
        # self.bic = None

    def score(self, y_act, y_fit, ndf, mdf):
        y_act = check_data_1d(y_act)
        y_fit = check_data_1d(y_fit)
        self._deviation_measures(y_act, y_fit)
        self._model_measures(y_act, y_fit, ndf, mdf)

    def summary(self, return_df=False):
        s = "*" * 80 + '\n'
        s += " Model Summary Statistics"
        s += ' - ' + self.identifier + '\n' if self.identifier is not None else "\n"
        s += "*" * 80 + '\n'
        s += 'Mean Error (ME)                  :  {:5.4f} \n'.format(self.me)
        s += 'Multiplicative Bias              :  {:5.4f} \n'.format(self.bias)
        s += 'Mean Abs Error (MAE)             :  {:5.4f} \n'.format(self.mae)
        s += 'Mean Abs Deviance Error (MAD)    :  {:5.4f} \n'.format(self.mad)
        s += 'Mean Abs Percentage Error(MAPE)  :  {:5.4f} \n'.format(self.mape)
        s += 'Root Mean Squared Error (RMSE)   :  {:5.4f} \n'.format(self.rmse)
        s += 'R-Squared                        :  {:5.4f} \n'.format(self.r2_val)
        s += 'Adj R-Squared                    :  {:5.4f} \n'.format(self.adjr2_val)
        s += 'F-Statistic                      :  {:5.4f} \n'.format(self.fstat[0])
        s += 'Prob (F-Statistic)               :  {:5.4f} \n'.format(self.fstat[1])
        s += 'Degree of Freedom - Residual     :  {:d} \n'.format(self.ndf)
        s += 'Degree of Freedom - Model        :  {:d} \n'.format(self.mdf)
        # s += 'Log Likelihood                   :  {:5.4f} \n'.format(self.loglik)
        # s += 'Akaike Info. Criterion (AIC)     :  {:5.4f} \n'.format(self.aic)
        # s += 'Bayesian Info. Criterion (BIC)   :  {:5.4f} \n'.format(self.bic)
        s += "*" * 80 + '\n'
        print(s)

        if return_df:
            return None


class BaseModelInterface:
    """
    Class defining unified interface for all Models
    """

    def __init__(self):
        self.model_name = None
        self.model_type = None
        self.error_type = None
        self.state = {}
        self.parameters = None
        self.ge_func = None
        self.confidence = None
        self.stats = None
        self.path = {}

    def plot_model(self):
        raise NotImplementedError
        # for series, data in self.path.items():
        #     if series == 'time':
        #         continue
        #     plt.plot(self.path['time'], data, label=series)
        # plt.legend(loc=0)
        # plt.grid()
        # plt.title("{} Model of {} Type".format(self.model_name, self.model_type))

    def plot_confidence(self):
        raise NotImplementedError
