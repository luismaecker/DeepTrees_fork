#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'deeptrees'
DESCRIPTION = 'AI-based delineation and analysis of individual tree crowns'
URL = 'https://codebase.helmholtz.cloud/ai-consultants-dkrz/DeepTrees'
AUTHOR = 'Caroline Arnold, Harsh Grover, Taimur Khan'
EMAIL = "taimur.khan@ufz.de"
REQUIRES_PYTHON = '>=3.10.0'
VERSION = None
LICENSE = "Prosperity Public License 3.0.0"
REQUIREMENTS = "requirements.txt"
EXCLUDES = ('tests', 'docs', 'images', 'build', 'dist')

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(here, REQUIREMENTS), encoding='utf-8') as f:
        REQUIRED = f.read().split('\n')
except:
    REQUIRED = []

# What packages are optional?
EXTRAS = {
    'test': ['pytest']
}

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Read version from package init
if not VERSION:
    with open(os.path.join(here, NAME, '__init__.py')) as f:
        for line in f.readlines():
            if "__version__" in line:
                VERSION = line.split('=')[-1].strip()[1:-1]
                print(VERSION, file=sys.stderr)

if VERSION is None:
    raise ValueError("Version is None")


def read_requirements():
    """
    Reads the requirements from the requirements.txt file.

    Returns:
        list: A list of requirements.
    """
    with open(REQUIREMENTS) as f:
        return f.read().splitlines()


def get_version():
    """
    Retrieves the version of the package.

    Returns:
        str: The version of the package.
    """
    if VERSION:
        return VERSION
    else:
        with open(os.path.join(here, NAME, '__version__.py')) as f:
            exec(f.read())
        return locals()['__version__']


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print(s)

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        author=AUTHOR,
        author_email=EMAIL,
        python_requires=REQUIRES_PYTHON,
        url=URL,
        packages=find_packages(exclude=EXCLUDES),
        # If your package is a single module, use this instead of 'packages':
        # py_modules=['mypackage'],

        # entry_points={
        #     'console_scripts': ['mycli=mymodule:cli'],
        # },
        install_requires=REQUIRED,
        extras_require=EXTRAS,
        include_package_data=True,
        license=LICENSE,
        classifiers=[
            # Trove classifiers
            # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
            # 'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: Implementation :: CPython',
            'Programming Language :: Python :: Implementation :: PyPy'
        ],
        # $ setup.py publish support.
        cmdclass={
            'upload': UploadCommand,
        },
)
##


##


##

