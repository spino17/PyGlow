import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyGlow",
    version="0.0.1",
    author="Bhavya",
    author_email="bhavyabhatt17@gmail.com",
    description="This package is an attempt to implement Keras like API functionalities on PyTorch backend.",
    long_description=long_description,
    url="https://github.com/spino17/PyGlow",
    install_requires=['numpy'],
    license='MIT',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)