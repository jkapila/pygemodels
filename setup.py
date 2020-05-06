from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pygemodels',
      version='0.0.1',
      description='Python library for Growth and Epidemiology Model Fitting Routines',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/jkapila/pygemodels',
      # project_urls={'Documentation': 'https://pygemodels.readthedocs.io'},
      author='Jitin Kapila',
      author_email='jitin.kapila@gmail.com',
      packages=['gemodels'],
      requires=['scipy', 'numpy', 'pandas'],
      # install_requires=['numpy>=1.16.3', 'tqdm>=4.31.1', 'scipy>=1.3.0'],
      python_requires='>=3.6',
      # test_suite='tests',
      license='MIT',
      zip_safe=False,
      include_package_data=True,
      )