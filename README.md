<p align="center">
  <img src='/docs/source/_static/PyGlow_complete_logo.jpg' width="600">
</p>

PyGlow is a Python package which attempts to implement Keras like API struture on PyTorch backend. It provides functionalities which supports information theoretic methods in deep learning. These methods are relevant for understanding neural network dynamics in information plane. The package is equipped with a number of state-of-the-art algorithms for estimating and calculating various kinds of information theoretic measures. The package also provides intensive support for information bottleneck based methods in deep learning. 



# What's new with PyGlow ?

PyGlow provide support for the following features of the 'IB theory of Deep Learning':

- Functionalities for experimental IB-based DNN analysis

- Test IB-based DNN performance bounds

- Flexible API structure to test theoretical ideas and hypothesis related to IB-theory or information theory in general

- IB-based training paradigms for DNN

- Flexible internal pipeline which allows for the implementation of custom user-defined dependence criterions or IB-based loss functions 

  

# Examples

Let's see with an example how you can really analyse the dynamics of DNNs using either in-built criterion or your own custom 'user-defined' criterion

```python
# importing glow module
import glow
from glow.layers import Dense, Dropout, Conv2d, Flatten
from glow.datasets import mnist
from glow.models import IBSequential
from glow.information_bottleneck.estimator import HSIC

# define hyperparameters
# hyperparameter
batch_size = 64
num_workers = 3
validation_split = 0.2
num_epochs = 10

# load the dataset
train_loader, val_loader, test_loader = mnist.load_data(
    batch_size=batch_size, num_workers=num_workers, validation_split=validation_split
)

# define your model
model = IBSequential(input_shape=(1, 28, 28), gpu=False, track_dynamics=False, save_dynamics=True)
model.add(Conv2d(filters=16, kernel_size=3, stride=1, padding=1))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
```

Yes the code structure looks exactly like Keras ! so that you can really try out new ideas of your own without worrying much about the code and debugging !

Now lets compile the model and attach HSIC criterion (more on this can be found on the [link](<https://wiki.math.uwaterloo.ca/statwiki/index.php?title=measuring_Statistical_Dependence_with_Hilbert-Schmidt_Norm>)) evaluator to it and finally train it ! To use your own custom built criterion visit this notebook [custom_criterion](<https://github.com/spino17/PyGlow/blob/master/docs/source/examples/Custom_criterion_class.ipynb>).

```python
# compile the model
model.compile(optimizer='SGD', loss='cross_entropy', metrics=['accuracy'])

# attach evaluator for criterion (HSIC in example) used while calculating coordinates of dynamics
model.attach_evaluator(HSIC(kernel='gaussian', gpu=True, sigma=5))

# train the model
model.fit_generator(train_loader, val_loader, num_epochs)
```

That's it ! you can now extract the evaluated dynamics from the model. For complete code checkout this [notebook](<https://github.com/spino17/PyGlow/blob/master/docs/source/examples/Analysing_dynamics_HS_Criterion.ipynb>).

Now that you have seen the exciting implications of PyGlow and with how much clean interface you can implement information theoretic methods for your own model, just head over to install it on your local machine and to really open the box of deep learning !

You can find more examples on either [docs page](http://pyglow.000webhostapp.com/jupyternotebooks.html) or related notebooks in github repo at [examples](<https://github.com/spino17/PyGlow/tree/master/docs/source/examples>).



# Requirements

Before installation there are some dependencies of PyGlow which needs to be taken care off for smooth installation.

PyGlow requires the following Python packages:

- NumPy, for basic numerical data structure
- PyTorch, for supporting backend engine
- tdqm, for progress bar functionalities



# Installation

## Installing dependencies

### PyTorch

PyGlow requires PyTorch backend, so to install it first run the following command (if already installed then skip this section).

```console
pip install torch
```
Now its time to install PyGlow on your system !

## Installing PyGlow

Currently the package is in development phase and can be installed from either Test PyPI or PyPI .

From TestPyPI
*************

```console
pip install -i https://test.pypi.org/simple/ PyGlow
```

From PyPI
*********
```console
   pip install PyGlow
```


# Documentation

Entire documentation is available on [Read the Docs](http://pyglow.000webhostapp.com/).



# Problems

If something unexpected happens while installing or using the package, do report it into the [issue tracker](https://github.com/spino17/PyGlow/issues).



# Contributing

PyGlow is a community project and we definitely look forward for contributions from all of the enthusiastic people out there ! 

While you contribute,  let us know that you are up for it by creating an issue or finding an existing one for the functionality or the feature you want to integrate into the core. Introduce your budding idea in the issue comments panel and discuss with us how are you planning to do it. 

Sending direct PR without following the above steps is highly discouraged and might end up with a rejected PR.



# Contact

Feel free to contact regarding any PyGlow related affair at bhavyabhatt17@gmail.com, we would love to receive queries from your side. 



# License

PyGlow is BSD-style licensed, including PyTorch license requirements as found in the LICENSE file.
