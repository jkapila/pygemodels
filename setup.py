from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements

def load_requirements(fname):
    reqs = parse_requirements(fname, session="test")
    return [str(ir.req) for ir in reqs]


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
      requires=['scipy', 'numpy','matplotlib'],
      install_requires=load_requirements("requirements.txt"),
      python_requires='>=3.6',
      # test_suite='tests',
      license='MIT',
      zip_safe=False,
      include_package_data=True,
      )