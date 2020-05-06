from gemodels import ModelStats
from gemodels.growth import GrowthModel
import numpy as np
import pandas as pd

if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b = [1, 2, 3, 4, 5, 6, 7, 9, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]

    assert len(a) == len(b)

    print('Generic Model Summary!')
    mod_stat = ModelStats()
    mod_stat.score(a, b, len(a) - 2, 2)
    mod_stat.summary()

    print('Fitting Logistic Model')

    a, b, c = 1.632,  0.5209, 0.0137

    # setting up an exponential decay function
    def decay(x, intercept, factor, exponent):
        return intercept - factor * np.exp(-exponent * x)

    # a function to generate exponential decay with gaussian noise
    def generate(intercept, factor, exponent):
        x = np.linspace(0.5, 500, num=100)
        y = decay(x, intercept, factor, exponent) + np.random.normal(loc=0, scale=0.05, size=100)
        return (x, y)

    # plot and generate some data
    np.random.seed(1)
    x, y = generate(a, b, c)
    X = np.vstack((x,y)).T
    print('Data has shape: ', X.shape)
    print('Data head: \n', X[:5,:])

    mod_growth = GrowthModel()
    mod_growth.fit(X)
    mod_growth.summary()
    steps = 10
    print('Making Prediction for steps: ',steps)
    print('Predictions: ',mod_growth.predict(steps))
    mod_growth.plot()

    url = 'https://apmonitor.com/che263/uploads/Main/stats_data.txt'
    data = pd.read_csv(url)
    X = data[['x', 'y']].values

    mod_growth = GrowthModel('exponential')
    mod_growth.fit(X)
    mod_growth.summary()
    mod_growth.plot(sigma=0.2)
