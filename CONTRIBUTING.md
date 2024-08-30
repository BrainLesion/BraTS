[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Contributing to BraTS

First off, thanks for taking the time to contribute! 🎉


## Contribute Code 
Fork the repository, clone it and implement your contribution.

**Setup:**
- We use [poetry](https://python-poetry.org/), make sure it is installed: `pip install poetry`
- Install dependencies by running: `poetry install`

**Requirements:**
- Our project follows the [black code style](https://github.com/psf/black). Make sure your code is formatted accordingly.
- Please add _meaningful_ docstring for your functions and annotate types
- Please add _meaningful_ tests for your contribution in `/tests` and make sure _all_ tests are passing by running `python -m pytest`



Once done, create a Pull Request to integrate the code into our project!
