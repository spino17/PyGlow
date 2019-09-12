Getting Started
===============

Overview
--------

Since the success of deep learning, where it outperforms almost every machine learning task including computer vision, speech recognition, natural language processing, representation learning and many more. Even after so many years after it first shone like a charm, there lies many subtle mysterious regarding theoretical aspects of deep learning which revolves around the questions dealing with generalization, memorization, regularization, hidden representations and many more such mind-boggling paradoxes. Many great researchers have put forward their views and perspectives on how to approach towards a theoretical framework for deep learning which has explanations and interpretations of what we observe in practice while training a deep learning model. One of the most promising ideas which is there for quite some time now is the application of information theory in deep learning using which researchers have shown interesting state-of-the-art results comparable to results from standard practices in deep learning. One such idea borrowed from information theory is ‘Information Bottleneck’ principle (IB) proposed by Prof. Naftali Tishby which give some hope for theoretically measure generalization and compression ability of a model. One of the key ideas of IB theory is its ability to define optimal representations even for intermediate hidden layers and this idea of regularizing intermediate representations have got much of the attention from all the researcher in the field. Since then many new measures and robust IB objective functions is proposed, all exploiting the above idea of regularization. PyGlow is the first library of its kind in the sense it provides a complete ecosystem for researchers working in this field by facilitating them with all the information theoretic methods they require while experimenting their ideas and hypothesis based on IB theory.


Intended Audience
-----------------

The intended audience for this package is really anyone who is willing to look beyond a traditional way of doing deep learning. The aim is to provide any deep learning enthusiast with a set of tools which she/he can use to open the black box of deep learning. From past two years, information theory have shown to give interesting insights into the theoretical studies of neural network dynamics. But no standard package existed before this which can really fill this gap of standard deep learning and this budding field of information theory in deep learning. There are researchers from all over the world publishing their awesome results in the field and now it's the perfect time for a library like PyGlow to get introduced itself to the researchers out there. Hence intended audience also include researchers which are or is interested to work in this field and this spirit of offering ease to them in their research work is reflected in the API structure of PyGlow. 
  

Installation
------------

Installing dependencies
.......................

PyTorch
*******
PyGlow requires PyTorch backend, so to install it first run the following command. ::

   pip install torch

Now its time to install PyGlow on your system !

Installing PyGlow
.................

Currently the package is in development phase and can be installed from either Test PyPI or PyPI .

From TestPyPI
*************
::

   pip install -i https://test.pypi.org/simple/ PyGlow

From PyPI
*********
::

   pip install PyGlow


Running your first code with PyGlow
-----------------------------------
Let's run your first PyGlow code.
First verify that it's installed successfully on your system by running the following command in the interpreter.

.. code-block:: python
  :linenos:
  
  import glow
  glow.__version__
>>> '0.1.6'

Now that you have verified the installation and have tried something yourself, head over to the examples section to make yourself comfortable with coding semantics starting with this example.


Contribute
----------

PyGlow is a community project, hence all contributions are more than
welcome!

Bug reporting
.............

Not only things break all the time, but also different people have different
use cases for the project. If you find anything that doesn't work as expected
or have suggestions, please refer to the `issue tracker`_ on GitHub.

.. _`issue tracker`: https://github.com/spino17/PyGlow/issues

Documentation
.............

Documentation can always be improved and made easier to understand for
newcomers. The docs are stored in text files under the `docs/source`
directory, so if you think anything can be improved there please edit the
files and proceed in the same way as with `code writing`_.

The Python classes and methods also feature inline docs: if you detect
any inconsistency or opportunity for improvement, you can edit those too.

Besides, the `wiki`_ is open for everybody to edit, so feel free to add
new content.

To build the docs, you must first create a development environment (see
below) and then in the ``docs/`` directory run:

 .. code:: bash

    $ cd docs
    $ make html

After this, the new docs will be inside ``build/html``. You can open
them by running an HTTP server:

 .. code:: bash

    $ cd build/html
    $ python -m http.server
    Serving HTTP on 0.0.0.0 port 8000 ...

And point your browser to http://0.0.0.0:8000.

Code writing
............

Code contributions are welcome! If you are looking for a place to start,
help us fixing bugs in einsteinpy and check out the `"easy" tag`_. Those
should be easier to fix than the others and require less knowledge about the
library.

.. _`"easy" tag`: https://github.com/spino17/PyGlow/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22

If you are hesitant on what IDE or editor to use, just choose one that
you find comfortable and stick to it while you are learning. People have
strong opinions on which editor is better so I recommend you to ignore
the crowd for the time being - again, choose one that you like :)

If you ask me for a recommendation, I would suggest Spyder (good for machine learning) 
or PyCharm (complete IDE, free and gratis, RAM-hungry) or vim (powerful editor, very 
lightweight, steep learning curve). Other people use Spyder, emacs, gedit, Notepad++,
Sublime, Atom...

You will also need to understand how git works. git is a decentralized
version control system that preserves the history of the software, helps
tracking changes and allows for multiple versions of the code to exist
at the same time. If you are new to git and version control, I recommend
following `the Try Git tutorial`_.

.. _`the Try Git tutorial`: https://try.github.io/

If you already know how all this works and would like to contribute new
features then that's awesome! Before rushing out though please make sure it
is within the scope of the library so you don't waste your time -
`email`_ us.

.. _`email`: bhavyabhatt17@gmail.com

If the feature you suggest happens to be useful and within scope, you will
probably be advised to create a new `wiki`_ page with some information
about what problem you are trying to solve, how do you plan to do it and
a sketch or idea of how the API is going to look like. You can go there
to read other good examples on how to do it. The purpose is to describe
without too much code what you are trying to accomplish to justify the
effort and to make it understandable to a broader audience.

.. _`wiki`: https://github.com/spino17/PyGlow/wiki 

All new features should be thoroughly tested, and in the ideal case the
coverage rate should increase or stay the same. Automatic services will ensure
your code works on all the operative systems and package combinations
PyGlow support - specifically, note that PyGlow is a Python 3 only
library.

Development environment
.......................

These are some succint steps to set up a development environment:

1. `Install git <https://git-scm.com/>`_ on your computer.
2. `Register to GitHub <https://github.com/>`_.
3. `Fork PyGlow <https://help.github.com/articles/fork-a-repo/>`_.
4. `Clone your fork <https://help.github.com/articles/cloning-a-repository/>`_.
5. Install it in development mode using
   :code:`pip install --editable /path/to/einsteinpy/[dev]` (this means that the
   installed code will change as soon as you change it in the download
   location).
6. Create a new branch.
7. Make changes and commit.
8. `Push to your fork <https://help.github.com/articles/pushing-to-a-remote/>`_.
9. `Open a pull request! <https://help.github.com/articles/creating-a-pull-request/>`_


Code Linting
............

To get the quality checks passed, the code should follow some standards listed below:

1. `Black <https://black.readthedocs.io/en/stable/>`_ for code formatting.
2. `isort <https://isort.readthedocs.io/en/latest/>`_ for imports sorting.
3. `mypy <http://mypy-lang.org/>`_ for static type checking.

