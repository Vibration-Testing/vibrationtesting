Vibration Testing:
==================

A package for signal processing, modal analysis, and model reduction and model updating

.. .. include:: <isonum.txt>
.. image:: https://badge.fury.io/py/vibrationtesting.png/
    :target: http://badge.fury.io/py/vibrationtesting

.. image:: https://travis-ci.org/Vibration-Testing/vibrationtesting.svg?branch=master
    :target: https://travis-ci.org/Vibration-Testing/vibrationtesting

.. image:: https://zenodo.org/badge/50037940.svg
    :target: https://zenodo.org/badge/latestdoi/50037940

Joseph C. Slater
----------------

Welcome to `Vibration Testing <http://Vibration-Testing.github.io/vibrationtesting/>`_.

For more information, please see the `documentation for the Python version <http://Vibration-Testing.github.io/vibrationtesting/>`_

Installation
------------

If you aren't familiar at all with Python, please see  `Installing Python <https://github.com/vibrationtoolbox/vibration_toolbox/blob/master/docs/Installing_Python.rst>`_.

Installation is made easy with ``pip`` (or ``pip3``), with releases as we have time while we try
to create a full first release. Much of it works already, but I certainly need
issue reports (on `github <http://github.com/Vibration-Testing/vibrationtesting>`_) when something is not working as it should.

To install::

  pip install --user vibrationtesting

where ``--user`` isn't necessary if you are using a locally installed version of Python such as `Anaconda <https://www.continuum.io/downloads>`_.

To run, I recommend you open a `Jupyter <https://jupyter.org>`_ notebook by using ``jupyter notebook`` and then type::

  import vibrationtesting as vt

For examples, see the `JupyterNotebooks folder <https://github.com/Vibration-Testing/vibrationtesting/tree/master/JupyterNotebooks>`_. (In flux- also look in doc/Tutorials for now) Some of these have interactive capabilities that are only apparent when you run them yourself instead of just looking at them on github. Unfortunately our organization of these still leaves a little to be desired.

Installation of current code
____________________________

The usage documentation is far behind the current code, while the reference is way ahead of the released code due to the `autodoc  <http://www.sphinx-doc.org/en/stable/ext/autodoc.html>`_ capability of `Sphinx  <http://www.sphinx-doc.org/en/stable/>`_. Especially as of 2017, the code is in flux. So is the documentation. Releases to `PyPI <https://pypi.python.org/pypi>`_ are far behind current status as stopping to deploy would cost more time that it is worth. I have the objective of releasing a first non-beta version at the end of May 2018, but even this cannot be promised.

If you wish to install the current version of the software, read the instructions in `Contributing.rst  <https://github.com/Vibration-Testing/vibrationtesting/blob/master/CONTRIBUTING.rst>`_

That should be it. Please note issues on the `issues tab  <https://github.com/Vibration-Testing/vibrationtesting/issues>`_ on GitHub.

Quick notes:
-------------

The `convention used for the signal processing  <http://python-control.readthedocs.io/en/latest/conventions.html#time-series-convention>`_ is that of the `python controls module  <http://python-control.readthedocs.io/en/latest/>`_.
