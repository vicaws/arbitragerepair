import os
from setuptools import setup

here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

setup(
    name='arbitragerepair',
    packages=['arbitragerepair'],
    description='Model-free algorithms of detecting and repairing spread, butterfly and calendar arbitrages in European option prices.',
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=[
        'cvxopt>=1.3.0',
        'numpy>=1.19.0',
        'pandas>=1.0.5'
    ],
    python_requies='>=3.8',
    url='https://github.com/vicaws/arbitragerepair',
    version='1.0.2',
    license='MIT',
    author='Victor Wang',
    author_email='wangsheng.victor@gmail.com',
    keywords=['option', 'arbitrage', 'quantitative finance', 'data cleansing', 'asset pricing'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ]
)
