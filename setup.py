import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="PyGlow",
    version="0.0.4",
    description="Information Theory of Deep Learning",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/spino17/PyGlow",
    author="Bhavya Bhatt",
    author_email="bhavyabhatt17@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["torch>=1.0.0", "numpy", "torchvision"],
    entry_points={
        "console_scripts": [
            "realpython=reader.__main__:main",
        ]
    },
)
