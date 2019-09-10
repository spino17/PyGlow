<img src='/docs/source/_static/PyGlow_logo.jpg' width="300" height="400">

PyGlow is a Python package which attempts to implement Keras like API functionalities on PyTorch backend with functionalities supporting information theoretic methods which are relevant for understanding neural network dynamics. The package is equipped with a number of state-of-the-art algorithms for estimating and calculating various kinds of information theoretic measures and quantities. Along with traditional dependence criterion for random variables like mutual information, PyGlow also have newer dependence criterion like HSIC criterion which have proved to be more robust in terms of 'Information Bottleneck Theory of Deep Learning'. 

# What's new with PyGlow ?

PyGlow provide support for the following features of the 'IB theory of Deep Learning':

- Functionalities for experimental IB-based DNN analysis
- Test IB-based DNN performance bounds
- Flexible API structure to test theoretical ideas and hypothesis related to IB-theory or information theory in general
- IB-based training paradigms of DNN
- Flexible internal pipeline which allows for the implementation of custom user-defined dependence criterions or IB-based loss functions 

# Documentation

Entire documentation is available on [Read the Docs](https://pyglow.readthedocs.io/en/latest/).

# Examples 

# Requirements

PyGlow requires the following Python packages:

- NumPy, for basic numerical data structure
- PyTorch, for supporting backend engine
- tdqm, for progress bar functionalities

# Installation

# Problems

If something unexpected happens while installing or using the package, do report it into the [issue tracker](https://github.com/spino17/PyGlow/issues).

# Contributing

PyGlow is a community project and we definitely look forward for contributions from all of the enthusiastic people out there ! 

While you contribute,  let us know that you are up for it by creating an issue or finding an existing one for the functionality or the feature you want to integrate into the core. Introduce your budding idea in the issue comments panel and discuss with us how are you planning to do it. 

Sending direct PR without following the above steps is highly discouraged and might end up with a rejected PR.

# Citing

# License

PyGlow is BSD-style licensed, including PyTorch license requirements as found in the LICENSE file.
