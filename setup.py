#!/usr/bin/env python

from setuptools import setup
import os
import sys

if sys.version_info < (3, 5):
    sys.exit('Sorry, Python < 3.5 is not supported.')
# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open('vibrationtesting/__init__.py', 'rb') as fid:
    for line in fid:
        line = line.decode('utf-8')
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break

download_url = ('https://github.com/Vibration-Testing/vibrationtesting/\
                blob/master/dist/vibrationtesting-' + version + '.whl')

setup(name='vibrationtesting',
      version=version,
      description=('Signal processing, modal analysis, plotting, and system\
                   identification for vibrating systems'),
      author=u'Joseph C. Slater',
      author_email='joseph.c.slater@gmail.com',
      url='https://github.com/Vibration-Testing/vibrationtesting',
      packages=['vibrationtesting'],
      package_data={'vibrationtesting': ['../readme.rst', 'data/*.mat'],
                    '': ['readme.rst']},
      long_description=read('readme.rst'),
      keywords=['vibration', 'mechanical engineering', 'testing',
                'civil engineering', 'modal analysis'],
      install_requires=['numpy', 'scipy', 'matplotlib'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest']
      )

# https://docs.python.org/3/distutils/setupscript.html#additional-meta-data
