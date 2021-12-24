"""Install setup.py for defect_detection."""
from setuptools import setup, find_packages


REQUIREMENTS = ['numpy',]

setup(
    name='py_cwt2d',
    version=0.1,
    description='Two Dimensional Continuous Wavelet Transform',
    author='Leon Archer',
    author_email='na',
    url='https://github.com/LeonArcher/py_cwt2d',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=REQUIREMENTS
)
