"""
Interface to Growth Model Training Routine
"""

from gemodels import BaseModelInterface, ModelStats, ModelError, StatError
from gemodels import check_data, check_data_1d
from .models import all_func
import numpy as np
from scipy.optimize import curve_fit, minimize, least_squares
from scipy.stats import t as statt, f as statf, chi2 as statx2, nbinom as statnb


class GrowthModel(BaseModelInterface):
    """
    Boiler plat for all growth models
    """

    def __init__(self, model='logistic', method='curve', method_params=dict(),
                 alter_strategy=None, valid_steps=0, confidence_criteria='one-student',
                 confidence_alpha=0.5, inverse=False):
        """
        Growth Models


        :param model:'logistic','richard','bass','chapmanrichard','gompretz','weibull
        :param method: 'curve','lsq','minloss', # stochastic in progress
        :param method_params: extra params while training
        :param alter_strategy: Manipulate data before fitting 'ma', 'dtrend' in progress
        :param valid_steps: data points to do validity on actual data.
        :param confidence_criteria:  'covariance'
        :param confidence_alpha:  float value to define confidence interval
        :param inverse: a flag for true and false.
        """

        super().__init__()
        self.model_name = "Growth"
        self.model_type = model.title()
        self._model = all_func[model]
        self.method = method
        self.method_params = method_params
        self._func = None
        self._pfunc = self._model['parametric']
        self.parameters = self._model['parameters']
        self.stats = ModelStats(name='{} {} Model'.format(self.model_type, self.model_name),
                                p_alpha=confidence_alpha)
        self.alter_strategy = alter_strategy
        self.inverse = inverse
        self.valid_steps = valid_steps
        self.conf_criteria = confidence_criteria


    def _alter_data(self, X):
        return X

    def _fit_curve(self, X):
        y, t = check_data(X)
        popt, pcov = curve_fit(self._func, t, y)
        self.cov = pcov
        self.parameters = self.parameters._make(popt)
        print('Curve Fitted on {} {} Model with Parameters'.format(
            self.model_type,self.model_name) , self.parameters)
        return y, self.predict(t)

    def _fit_linear(self, X):
        y, t = check_data(X)
        opt = []
        self.parameters._make(opt)
        return y, self.predict(t)

    def _fit_stochastic(self, X):
        y, t = check_data(X)
        self.parameters._make([])
        return y, self.predict(t)

    def _fit_minimize(self, X, **kwargs):
        y, t = check_data(X)

        opt = []
        self.parameters._make(opt)
        return y, self.predict(t)

    def fit(self, X, **model_args):

        if self.method == "curve":
            self._func = self._model['curve']
            y_act, y_fit = self._fit_curve(X)

        elif self.method == "linear":
            self._func = self._model['curve']
            y_act, y_fit = self._fit_linear(X)

        elif self.method == "minimize":
            self._func = self._model['curve']
            y_act, y_fit = self._fit_minimize(X, **model_args)

        elif self.method == "stochastic":
            self._func = self._model['curve']
            y_act, y_fit =  self._fit_stochastic(X)

        else:
            raise ModelError('Not a Valid Method for fitting')

        self.stats.score(y_act=y_act, y_fit=y_fit,
                         ndf=len(y_act) - self._model['df_model'] - 1,
                         mdf=self._model['df_model'])

    def summary(self):
        self.stats.summary(return_df=False)

    def predict(self, steps):
        if isinstance(steps,int):
            # This is crude as of now need better methods
            t_steps = self.stats.ndf + self.stats.mdf + 1
            steps = np.arange(t_steps+1,t_steps+steps+1,dtype=np.int8)
        else:
            steps = check_data_1d(steps)
        return self._pfunc(self.parameters, steps)

    def plot_fit(self, X=None, confidence=True):
        pass

    def plot_forecast(self):
        pass
