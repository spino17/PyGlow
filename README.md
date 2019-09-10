<img src='/docs/source/_static/PyGlow_logo.jpg' width="200">

PyGlow is a Python package which attempts to implement Keras like API functionalities on PyTorch backend with functionalities supporting information theoretic methods which are relevant for understanding neural network dynamics. The package is equipped with a number of state-of-the-art algorithms for estimating and calculating various kinds of information theoretic measures and quantities. Along with traditional dependence criterion for random variables like mutual information, PyGlow also have newer dependence criterion like HSIC criterion which have proved to be more robust in terms of 'Information Bottleneck Theory of Deep Learning'. 



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

Now lets compile the model and attach HSIC criterion (more on this can be found on the [link]()) evaluator to it and finally train it ! to use your own custom built criterion visit this notebook [custom_criterion](<https://github.com/spino17/PyGlow/blob/master/docs/source/examples/Custom_criterion_class.ipynb>).

```python
# compile the model
model.compile(optimizer='SGD', loss='cross_entropy', metrics=['accuracy'])

# attach evaluator for criterion (HSIC in example) used while calculating coordinates of dynamics
model.attach_evaluator(HSIC(kernel='gaussian', gpu=True, sigma=5))

# train the model
model.fit_generator(train_loader, val_loader, num_epochs)
```

That's it ! you can now extract the evaluated dynamics from the model. For details on this just visit more awesome examples related to information theory of deep learning at [example](<https://github.com/spino17/PyGlow/tree/master/docs/source/examples>). 

Now that you have seen the exciting implications of PyGlow and with how much clean interface you can implement information theoretic methods just head over to install on your local machine and really unbox the deep learning !

You can find more example on either docs page at ... or related notebooks in github repo at [examples](<https://github.com/spino17/PyGlow/tree/master/docs/source/examples>).



# Requirements

Before installation there are some dependencies of PyGlow which needs to be taken care off for smooth installation.

PyGlow requires the following Python packages:

- NumPy, for basic numerical data structure
- PyTorch, for supporting backend engine
- tdqm, for progress bar functionalities



# Installation

### Installing PyTorch

PyGlow requires PyTorch backend so to install it first run the following command



Now its time to install PyGlow on your system and really unbox the deep learning !

### Installation of PyGlow

Currently the package is in development phase and is only available on Test PyPI .

```console
pip install -i https://test.pypi.org/simple/ PyGlow
```



# Documentation

Entire documentation is available on [Read the Docs](https://pyglow.readthedocs.io/en/latest/).



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
