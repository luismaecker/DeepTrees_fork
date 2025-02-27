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
DESCRIPTION = 'Tree crown segmentation and analysis in remote sensing imagery with PyTorch'
URL = 'https://codebase.helmholtz.cloud/taimur.khan/DeepTrees'
AUTHOR = 'Taimur Khan, Caroline Arnold, Harsh Grover'
EMAIL = "taimur.khan@ufz.de"
REQUIRES_PYTHON = '>=3.10.0'
VERSION = None
LICENSE = "MIT"
REQUIREMENTS = "requirements.txt"
EXCLUDES = ('tests', 'docs', 'images', 'build', 'dist')

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
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Read version from deeptrees/__init__.py if not provided
if not VERSION:
    with open(os.path.join(here, NAME, '__init__.py'), encoding='utf-8') as f:
        for line in f:
            if "__version__" in line:
                VERSION = line.split('=')[-1].strip().strip('"').strip("'")
                print(f"Found version in __init__.py: {VERSION}", file=sys.stderr)

# Function that checks GitLab's CI_COMMIT_TAG or falls back
def get_version():
    """
    Returns the version from GitLab tag (CI_COMMIT_TAG) if set,
    else falls back to the version parsed above from __init__.py.
    """
    env_version = os.getenv("CI_COMMIT_TAG")
    if env_version:
        # If the tag is "v1.2.3", strip the leading 'v'
        if env_version.startswith("v"):
            env_version = env_version[1:]
        return env_version
    return VERSION

# Final check in case nothing is set:
final_version = get_version()
if not final_version:
    raise ValueError("Could not determine the version. "
                     "Neither __init__.py nor CI_COMMIT_TAG provided it.")

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
        os.system('git tag v{0}'.format(final_version))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=final_version,
    description=DESCRIPTION,
    keywords='tree-crowns, pytorch, remote-sensing, deep-learning, semantic-segmentation, vegetation-ecology, instance-segmentation, active-learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=EXCLUDES),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    cmdclass={
        'upload': UploadCommand,
    },
)