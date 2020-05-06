"""
Interface to Growth Model Training Routine
"""

from gemodels import BaseModelInterface, ModelStats, ModelError, StatError
from gemodels import check_data, check_data_1d
from .models import all_func
import numpy as np
from scipy.optimize import curve_fit, minimize, least_squares
from matplotlib import pyplot as plt
from scipy.stats import t as statt, f as statf, chi2 as statx2, nbinom as statnb


class GrowthModel(BaseModelInterface):
    """
    Boiler plat for all growth models
    """

    def __init__(self, model='logistic', method='curve', method_params=dict(),
                 alter_strategy=None, valid_steps=0, confidence_criteria='one-student',
                 confidence_alpha=0.05, saddle_tol=1e-4, inverse=False, copy=True):
        """
        Growth Models


        :param model:'logistic','richard','bass','chapmanrichard','gompretz','weibull
        :param method: 'curve','lsq','minloss', # stochastic in progress
        :param method_params: extra params while training
        :param alter_strategy: Manipulate data before fitting 'ma', 'dtrend' in progress
        :param valid_steps: data points to do validity on actual data.
        :param confidence_criteria:  'covariance'
        :param confidence_alpha:  float value to define confidence interval
        :param saddle_tol: Tolerance to find the saddle point / stable state of the curve
        :param inverse: a flag for true and false.
        :param copy: Copy data with the model object
        """

        super().__init__()
        self.model_name = "Growth"
        self.model_type = model.title()
        self._model = all_func[model]
        self.method = method
        self.method_params = method_params
        self._func = None
        self._pfunc = self._model['parametric']
        self._parameters = self._model['parameters']
        self.parameters = None
        self.parameters_std = None
        self.stats = ModelStats(name='{} {} Model'.format(self.model_type, self.model_name),
                                p_alpha=confidence_alpha)
        self.alter_strategy = alter_strategy
        self.inverse = inverse
        self.valid_steps = valid_steps
        self.conf_criteria = confidence_criteria
        self.saddle_tol = saddle_tol
        self.state_flag = copy

    def _alter_data(self, y, t):
        # todo: implement moving average and all here
        if self.alter_strategy is None:
            return y, t

    def _get_saddle_point(self):
        # todo: make this
        return 0

    def _get_data(self, X=None, use_alter=True):
        if X is None and self.state_flag:
            X = self.state
        y, t = check_data(X)
        if use_alter:
            y, t = self._alter_data(y, t)
        return y, t

    # def __repr__(self):
    #     # todo: make this

    def _fit_curve(self, X, **kwargs):
        y, t = self._get_data(X)
        opt, covar_mat = curve_fit(self._func, t, y)

        # setting optimal parameters
        self.parameters = self._parameters._make(opt)

        # getting covariance based standard deviations
        sigma_ab = np.sqrt(np.diagonal(covar_mat))
        self.parameters_std = self._parameters._make(sigma_ab)

        print('Curve Fitted on {} {} Model with Parameters'.format(
            self.model_type, self.model_name), self.parameters)

        return y, self._pfunc(t, self.parameters)

    def _fit_linear(self, X, **kwargs):
        y, t = self._get_data(X)
        opt = []
        self.parameters = self._parameters._make(opt)
        return y, self.predict(t)

    def _fit_stochastic(self, X, **kwargs):
        y, t = self._get_data(X)
        opt = []
        self.parameters = self._parameters._make(opt)
        return y, self.predict(t)

    def _fit_minimize(self, X, **kwargs):
        y, t = self._get_data(X)
        opt = []
        self.parameters = self._parameters._make(opt)
        return y, self.predict(t)

    def fit(self, X, **model_args):
        """

        :param X:
        :param model_args:
        :return:
        """

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
            y_act, y_fit = self._fit_stochastic(X)

        else:
            raise ModelError('Not a Valid Method for fitting')

        self.stats.score(y_act=y_act, y_fit=y_fit,
                         ndf=len(y_act) - self._model['df_model'] + 1,
                         mdf=self._model['df_model'])
        if self.state_flag:
            self.state = X

    def summary(self):
        self.stats.summary(return_df=False)

    def _get_steps(self, steps, use_data=False, smoothed=False, breaks=100):
        """
        Step formulation, checking and smoothening
        :param steps: integer, list or 1D numpy array
        :param use_data: Use the exsisting data and add steps with them
        :param smoothed: To smoothed the steps or not
        :param breaks:
        :return:
        """

        if use_data and self.state_flag:
            _, steps = self._get_data()
            if smoothed:
                breaks = breaks if len(steps) < 0.75 * breaks else int(2 * len(steps))
                steps = np.linspace(int(0.95 * np.min(steps)), int(1.05 * np.max(steps)), breaks)
            return steps
        elif use_data:
            raise ModelError('Data is not stored with the model. Flag \'use_data\' wont work!')

        if self.state_flag: # based on value of data
            _ , t = self._get_data()
            t_steps = int(np.max(t))
        else: # based on degree of freedoms
            t_steps = self.stats.ndf + self.stats.mdf - 1

        if isinstance(steps, int) and smoothed:
            # This is crude as of now need better methods
            steps = np.linspace(t_steps + 1, t_steps + steps + 1, breaks)
        elif isinstance(steps,int) and not smoothed:
            steps = np.arange(t_steps + 1, t_steps + steps + 1)
        elif (isinstance(steps, list) or isinstance(steps, tuple)) and len(steps) == 2:
            steps = np.linspace(steps[0], steps[1], breaks)
        elif smoothed:
            breaks = breaks if len(steps) < 0.75 * breaks else int(2 * len(steps))
            steps = np.linspace(int(0.95 * np.min(steps)), int(1.05 * np.max(steps)), breaks)
        else:
            steps = check_data_1d(steps)

        return steps

    def predict(self, steps, response=False, sigma=1.96, breaks=100):

        steps = self._get_steps(steps, breaks=breaks)
        y_fit = self._pfunc(steps, self.parameters)
        if response:
            params = [self.parameters, self.parameters_std]
            uparam = self._parameters(*map(lambda x: x[0] + sigma * x[1], zip(*params)))
            lparam = self._parameters(*map(lambda x: x[0] - sigma * x[1], zip(*params)))
            fit_upper = self._pfunc(steps, uparam)
            fit_lower = self._pfunc(steps, lparam)
            return y_fit, fit_upper, fit_lower
        return y_fit

    def plot(self, title=None, new_data=None, plot_range=None, confidence=True, confidence_band=True, sigma=1.96,
             breaks=100, fig_size=(10, 7)):

        title = title if title is not None else 'Estimated {} {} Model'.format(self.model_type, self.model_name)
        try:
            y_act, t = self._get_data(new_data)
        except Exception as e:
            raise ModelError('No data to make a plot on or Data is not in right format! Aborting! Error:\n', e)

        # Confidence level
        alpha = int(100 - self.stats.confidence_alpha * 100)

        # plotting actual data
        # plt.figure(figsize=fig_size, dpi=300)
        plt.scatter(t, y_act, s=3, label='Data')

        # print("Actual Steps: ", t)
        # getting smoothed breaks
        if plot_range is None:
            plot_range = t
        t_smoothed = self._get_steps(plot_range, smoothed=True, breaks=breaks)
        y_fit = self._pfunc(t_smoothed, self.parameters)

        # print("Smooth Steps: ", t_smoothed)

        # plot the regression
        plt.plot(t_smoothed, y_fit, c='black',
                 label='{} {} Model'.format(self.model_type, self.model_name))

        if confidence:
            params = [self.parameters, self.parameters_std]
            uparam = self._parameters(*map(lambda x: x[0] + sigma * x[1], zip(*params)))
            lparam = self._parameters(*map(lambda x: x[0] - sigma * x[1], zip(*params)))
            fit_upper = self._pfunc(t_smoothed, uparam)
            fit_lower = self._pfunc(t_smoothed, lparam)

            plt.plot(t_smoothed, fit_lower, c='orange', label='{}% Confidence Region'.format(alpha))
            plt.plot(t_smoothed, fit_upper, c='orange')

        if confidence_band:
            lpb, upb = confidence_band_t(func=self._pfunc, params=self.parameters,
                                         y_act=y_act, t=t,
                                         t_breaks=t_smoothed,
                                         alpha=self.stats.confidence_alpha)

            plt.plot(t_smoothed, lpb, 'k--', label='{}% Prediction Band'.format(alpha))
            plt.plot(t_smoothed, upb, 'k--')

        plt.ylabel('Estimated Values')
        plt.xlabel('Data Steps')
        plt.title(title)
        plt.legend(loc='best')

        # save and show figure
        plt.savefig('{}.png'.format(title))
        plt.show()

    def plot_forecast(self, steps, plot_range=None, title=None, use_trianing=True,
                      confidence=True, return_forecast=False, sigma=1.96, fig_size=(10, 7)):
        # plt.figure(figsize=fig_size, dpi=300)
        title = title if title is not None else 'Estimated {} {} Model'.format(self.model_type, self.model_name)
        steps = self._get_steps(steps, use_data=use_trianing)

        # Confidence level
        alpha = int(100 - self.stats.confidence_alpha * 100)
        res = self.predict(steps, response=confidence, sigma=sigma)
        if confidence:
            plt.plot(steps, res[0], 'black', label='Forecast Values')
            plt.plot(steps, res[1], 'k--', label='{}% Prediction Band'.format(alpha))
            plt.plot(steps, res[2], 'k--')
        else:
            plt.plot(steps, res, 'black', label='Forecast Values')

        plt.ylabel('Estimated Values')
        plt.xlabel('Data Steps')
        plt.title(title)
        plt.legend(loc='best')
        plt.show()
        if return_forecast:
            return res
