User Guide
==========


Standard Models
---------------

.. automodule:: network

.. autoclass:: Network
    :members:

Sequential
..........

.. autoclass:: Sequential
    :members:

IBSequential
............

.. autoclass:: IBSequential
    :members:


'Models without Back-prop'
--------------------------

.. automodule:: hsic

.. autoclass:: HSIC
    :members:

HSICSequential
..............

.. autoclass:: HSICSequential
    :members:


Layers
------

.. automodule:: layer
    :members:

Core
....

.. automodule:: core

.. autoclass:: Dense
    :show-inheritance:

.. autoclass:: Dropout
    :show-inheritance:

.. autoclass:: Flatten
    :show-inheritance:



Convolutional
.............

.. automodule:: convolutional

.. autoclass:: Conv1d
    :show-inheritance:

.. autoclass:: Conv2d
    :show-inheritance:

.. autoclass:: Conv3d
    :show-inheritance:

Normalization
.............

.. automodule:: normalization

.. autoclass:: BatchNorm1d
    :show-inheritance:

.. autoclass:: BatchNorm2d
    :show-inheritance:

.. autoclass:: BatchNorm3d
    :show-inheritance:

Pooling
.......

.. automodule:: pooling

1-D
***

.. autoclass:: MaxPool1d
    :show-inheritance:

.. autoclass:: AvgPool1d
    :show-inheritance:

2-D
***

.. autoclass:: MaxPool2d
    :show-inheritance:

.. autoclass:: AvgPool2d
    :show-inheritance:

3-D
***

.. autoclass:: MaxPool3d
    :show-inheritance:

.. autoclass:: AvgPool3d
    :show-inheritance:


HSIC
....

.. automodule:: hsic_output

.. autoclass:: HSICoutput
    :show-inheritance:


Information Bottleneck
----------------------

Estimator
.........

.. automodule:: estimator

.. autoclass:: Estimator
    :members:

.. autoclass:: HSIC
    :members:


Preprocessing
-------------

Data Loading and Generation
...........................

.. automodule:: data_generator
    :members:

