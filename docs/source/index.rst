.. PyGlow documentation master file, created by
   sphinx-quickstart on Tue Sep 10 10:31:49 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyGlow's documentation!
==================================

PyGlow is a Python package which attempts to implement Keras like API functionalities on PyTorch backend with functionalities supporting information theoretic methods which are relevant for understanding neural network dynamics. The package is equipped with a number of state-of-the-art algorithms for estimating and calculating various kinds of information theoretic measures and quantities. Along with traditional dependence criterion for random variables like mutual information, PyGlow also have newer dependence criterion like HSIC criterion which have proved to be more robust in terms of 'Information Bottleneck Theory of Deep Learning'.


What's new with PyGlow ?
------------------------
PyGlow provide support for the following features of the 'IB theory of Deep Learning':

* Functionalities for experimental IB-based DNN analysis
* Test IB-based DNN performance bounds
* Flexible API structure to test theoretical ideas and hypothesis related to IB-theory or information theory in general
* IB-based training paradigms of DNN
* Flexible internal pipeline which allows for the implementation of custom user-defined dependence criterions or IB-based loss functions

Installation
------------
Before any coding lets first install the package !
Follow the instruction on 

Example
-------
Let's see a code snippet which tracks the input-hidden-label segment (called dynamics segment) for each batch of each epoch and calculates the HSIC criterion between each hidden layer, input and label of a dynamics segment which is for each layer of each batch of each epoch. One thing that you will find interesting is the resemblance with Keras code structure of defining the model. PyGlow closely follows the API structure of Keras and provides with an 'easy to implement' pipeline so that you can really work on your ideas rather than debugging ;) 

.. code-block:: python
  :linenos:
  
  # importing glow modules
  from glow.models import IBSequential
  from glow.layers import Dense, Dropout, Conv2d, Flatten, HSICoutput
  from glow.datasets import mnist
  from glow.information_bottelneck.estimator import HSIC
  
  # declaring hyperparameters
  batch_size = 64
  num_workers = 3
  validation_split = 0.2
  pre_num_epochs = 5
  post_num_epochs = 5

  # load the dataset
  train_loader, val_loader, test_loader = mnist.load_data(
      batch_size=batch_size, num_workers=num_workers, validation_split=validation_split
  )

  # define the IB model - tracks the information plane coordinates
  model = IBSequential(input_shape=(1, 28, 28), gpu=True, 
                                   track_dynamics=True, save_dynamics=True)
  model.add(Conv2d(filters=16, kernel_size=3, stride=1, padding=1))
  model.add(Flatten())
  model.add(Dropout(0.4))
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(10, activation='softmax'))

  # compile the model
  model.compile(optimizer='SGD', loss='cross_entropy', metrics=['accuracy'])

  # attach evaluator - Tracks the dynamics and calculate coordinates 
  # according to HSIC criterion
  model.attach_evaluator(HSIC(kernel='gaussian', gpu=True, sigma=5))
  
  # train the model
  model.fit_generator(train_loader, val_loader, num_epochs)



Guide
-----

.. toctree::
   :maxdepth: 2
   
   installation
   license



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
