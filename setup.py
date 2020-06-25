from setuptools import setup
from setuptools import find_packages

install_requires = ['matplotlib>=3.0.2',
                    'numpy>=1.18.1',
                    'plotly>=4.6.0',
                    'scikit-image>=0.16.2',
                    'scikit-learn>=0.21.3',
                    'seaborn>=0.10.0',
                    'tensorflow>=1.2.0',
                    'tensorflow_datasets']

setup(name='matilda',
      version='0.0.1',
      packages=find_packages(),
      include_package_data=False,
      install_requires=install_requires)
