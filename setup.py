# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module.

Reference:
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import os
import sys

from setuptools import setup, find_packages, Command


class Formatter(Command):
    """Cmdclass for `python setup.py format`.

    Args:
        Command (distutils.cmd.Command):
            Abstract base class for defining command classes.
    """

    description = 'format the code using yapf'
    user_options = []

    def initialize_options(self):
        """Set default values for options that this command supports."""
        pass

    def finalize_options(self):
        """Set final values for options that this command supports."""
        pass

    def run(self):
        """Fromat the code using yapf."""
        errno = os.system('python3 -m yapf --in-place --recursive --exclude .git --exclude .eggs .')
        sys.exit(0 if errno == 0 else 1)


class Linter(Command):
    """Cmdclass for `python setup.py lint`.

    Args:
        Command (distutils.cmd.Command):
            Abstract base class for defining command classes.
    """

    description = 'lint the code using yapf and flake8'
    user_options = []

    def initialize_options(self):
        """Set default values for options that this command supports."""
        pass

    def finalize_options(self):
        """Set final values for options that this command supports."""
        pass

    def run(self):
        """Lint the code with yapf, mypy, and flake8."""
        errno = os.system(
            ' && '.join([
                'python3 -m yapf --diff --recursive --exclude .git --exclude .eggs .',
                'python3 -m flake8',
            ])
        )
        sys.exit(0 if errno == 0 else 1)


class Tester(Command):
    """Cmdclass for `python setup.py test`.

    Args:
        Command (distutils.cmd.Command):
            Abstract base class for defining command classes.
    """

    description = 'test the code using pytest'
    user_options = []

    def initialize_options(self):
        """Set default values for options that this command supports."""
        pass

    def finalize_options(self):
        """Set final values for options that this command supports."""
        pass

    def run(self):
        """Run pytest."""
        errno = os.system('python3 -m pytest -v --cov=msamp --cov-report=xml --cov-report=term-missing tests/')
        sys.exit(0 if errno == 0 else 1)


setup(
    name='msamp',
    url='https://github.com/Azure/MS-AMP',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'colorlog>=6.7.0',
    ],
    extras_require={
        'dev': ['pre-commit>=2.10.0'],
        'test': [
            'flake8-docstrings>=1.5.0',
            'flake8-quotes>=3.2.0',
            'flake8>=3.8.4',
            'mypy>=0.800',
            'pydocstyle>=5.1.1',
            'pytest-cov>=2.11.1',
            'pytest-subtests>=0.4.0',
            'pytest>=6.2.2',
            'types-pyyaml',
            'vcrpy>=4.1.1',
            'yapf>=0.30.0',
        ],
    },
    cmdclass={
        'format': Formatter,
        'lint': Linter,
        'test': Tester,
    },
    project_urls={
        'Source': 'https://github.com/Azure/MS-AMP',
        'Tracker': 'https://github.com/Azure/MS-AMP/issues',
    },
)
