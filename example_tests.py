from gemodels import ModelStats
from gemodels.growth import GrowthModel
import pandas as pd

if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b = [1, 2, 3, 4, 5, 6, 7, 9, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]

    assert len(a) == len(b)

    print('Generic Model Summary!')
    mod_stat = ModelStats()
    mod_stat.score(a, b, len(a) - 2, 2)
    mod_stat.summary()

    print('Fitting logistic _model')
    url = 'https://apmonitor.com/che263/uploads/Main/stats_data.txt'
    data = pd.read_csv(url)
    X = data.values

    mod_growth = GrowthModel()
    mod_growth.fit(X)
    mod_growth.summary()

    max_x = int(data['x'].max().tolist())
    steps = [i+max_x for i in range(1,11)]
    print('Making Prediction for steps: ',steps)
    print('Predictions: ',mod_growth.predict(steps))
