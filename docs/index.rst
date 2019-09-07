.. PyGlow documentation master file, created by
   sphinx-quickstart on Thu Sep  5 14:28:52 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyGlow's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

PyGlow
========

PyGlow package is an attempt to implement Keras like API functionalities on PyTorch backend with functionalities supporting information theoretic methods which are relevant for understanding neural network dynamics.

Look how easy it is to use:

    import glow
    # Get your stuff done

Features
--------

- Track the dynamics of the training process.
- Estimate information theoretic measures that explains generalization and compression.
- Attach your custom evaluators for intermediate layers to train neural networks.
- State-of-the-art mutual information (other criterion) estimators to give accurate bounds.
- Easy to implement and experiment with information theoretic methods in deep learning.
- Ease of Keras with Backend engine on PyTorch. 

Installation
------------

Install PyGlow by running:

    pip install -i https://test.pypi.org/simple/ PyGlow

Contribute
----------

- Issue Tracker: github.com/spino17/PyGlow/issues
- Source Code: github.com/spino17/PyGlow

Support
-------

If you are having issues, please let us know.
E-mail us at: bhavyabhatt17@gmail.com

License
-------

PyGlow is BSD-style licensed, including PyTorch license requirements as found in the LICENSE file.
