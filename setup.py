# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module.

Reference:
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import os
import sys

from setuptools import setup, Command


class Formatter(Command):
    """Cmdclass for `python setup.py format`.
    test test
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


setup(cmdclass={
    'format': Formatter,
    'lint': Linter,
    'test': Tester,
})
