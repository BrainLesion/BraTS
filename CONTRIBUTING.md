[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPLv3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/Agpl-3.0)

# Contributing to BraTS

First off, thanks for taking the time to contribute! 🎉

## Feature Requests
We welcome new feature requests! Please open an issue [here](https://github.com/BrainLesion/BraTS/issues) and describe:

- The problem you're facing.
- A possible solution or feature.

## Reporting Bugs

If you find a bug, please open an issue [here](https://github.com/BrainLesion/BraTS/issues) and include:

- A clear, descriptive title.
- Steps to reproduce the bug.
- Your environment (OS, Python version, etc.).

## Contribute Code 
Fork the repository, clone it and implement your contribution.

**Setup:**
- We use [poetry](https://python-poetry.org/), make sure it is installed: `pip install poetry`
- Install dependencies by running: `poetry install`

**Requirements:**
- Our project follows the [black code style](https://github.com/psf/black). Make sure your code is formatted correctly.
- Please add _meaningful_ docstring for your functions and annotate types
- Please add _meaningful_ tests for your contribution in `/tests` and make sure _all_ tests are passing by running `python -m pytest`



Once done, create a Pull Request to integrate the code in to our project!

