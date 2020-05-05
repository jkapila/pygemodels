# """
# This is where all growths models are
# """

from gemodels import BaseModelInterface


class EpiModel(BaseModelInterface):

    def __init__(self):
        super().__init__()
        self.model_name = "EPI Growth Model"

    def fit(self, X):
        raise NotImplementedError

    def fit_linear(self,X):
        raise NotImplementedError

    def summary(self):
        pass

    def predict(self):
        raise NotImplementedError

    def plot(self,X=None):
        pass