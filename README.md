# pygemodels
=========


<!--.. image:: https://travis-ci.org/lebigot/uncertainties.svg?branch=master-->
<!--   :target: https://travis-ci.org/lebigot/uncertainties-->
<!--.. image:: https://ci.appveyor.com/api/projects/status/j5238244myqx0a0r?svg=true-->
<!--   :target: https://ci.appveyor.com/project/lebigot/uncertainties-->
<!--.. image:: https://codecov.io/gh/lebigot/uncertainties/branch/master/graph/badge.svg-->
<!--   :target: https://codecov.io/gh/lebigot/uncertainties/-->
<!--   -->
<!--.. image:: https://readthedocs.org/projects/uncertainties-python-package/badge/?version=latest-->
<!--   :target: http://uncertainties-python-package.readthedocs.io/en/latest/?badge=latest-->
<!--.. image:: https://img.shields.io/pypi/v/uncertainties.svg-->
<!--   :target: https://pypi.org/project/uncertainties/-->


###  Python library for Growth and Epidemiology Model Fitting Routines

***The Need : The main goal here is to estimate the models based on data
availability and inferences on statistical tests.***


###  Implementations
========

####  Implementation for growth model includes:
1. Logistic Growth Model
2. Richard Curve
3. Gompertz Curve
4. Bass Diffusion CDF
5. Weibull

In Pipeline
1.  Generalized Logistic Growth Model
2.  Generalized Richard Curve

####  Implementation for Epidemiology model includes:
1. SIR
2. SIS
3. SIRS
4. SEIR
5. SEQIAHR

*Above are in Pipeline and not implemented yet.*

***These Models can be trained individually or via Interfaces***

### Examples
===========

Growth Models can be trained as:

```python
from gemodels.growth import GrowthModel

mod_growth = GrowthModel()
mod_growth.fit(X)
mod_growth.summary()
print('Predictions: ',mod_growth.predict(10))
mod_growth.plot()

```
Epi Models can be trained as:
```python
from gemodels.epi import EpiModel

mod_epi = EpiModel()
mod_epi.fit(X)
mod_epi.summary()
print('Predictions: ',mod_epi.predict(10))
mod_epi.plot()

```

**Note: This project is under heavy development. Routines can break.
Please use with discretion and caution.**

### Disclaimer:
===========

Lot of code for this project has been inspired by following projects:
1. [R EpiModel](https://github.com/statnet/EpiModel)
2. [Python epimodel by fccoelho](https://github.com/fccoelho/epimodels)
3. [R growthrates](https://github.com/tpetzoldt/growthrates)
4. [R PVAClone](https://github.com/psolymos/PVAClone)
5. [Python EpiGrass](https://github.com/fccoelho/epigrass)
6. [Python EpiStochastic](https://github.com/fccoelho/EpiStochModels)
7. [Python EpiMode 2](https://github.com/kuperov/epimodel)



