=============================================
 The Engineering Vibration Toolbox for Python
=============================================


.. .. include:: <isonum.txt>
.. image:: https://badge.fury.io/py/vibrationtesting.png/
    :target: http://badge.fury.io/py/vibrationtesting

.. image:: https://travis-ci.org/Vibration-Testing/vibrationtesting.svg?branch=master
    :target: https://travis-ci.org/Vibration-Testing/vibrationtesting

.. .. image:: https://img.shields.io/pypi/v/vibration_toolbox.svg
    :target: https://img.shields.io/pypi/v/vibration_toolbox

.. #image:: https://coveralls.io/repos/vibrationtoolbox/vibration_toolbox/badge.png?branch=master
..  #:target: https://coveralls.io/r/vibrationtoolbox/vibration_toolbox


Joseph C. Slater

This is the software repository for the Vibration Toolbox, not the `online manual <http://Vibration-Testing.github.io/vibrationtesting/>`_.

Welcome to `Vibration Testing <http://Vibration-Testing.github.io/vibrationtesting/>`_.
Originally written for `Matlab <http://www.mathworks.com>`_\®, this `Python <http://python.org>`_ version is a completely new design build for modern applications. This is an *applied* set of codes intended primarily for
researchers that accompanies the text by the author (unpublished). You may find them useful for application, but that
isn't the intent.

For more information, please see the `documentation for the Python version <http://Vibration-Testing.github.io/vibrationtesting/>`_ but please excuse that it is all still under development. Such documentation has never existed for the other ports of the toolbox so this is taking some time. We don't need feedback at this time, but we will take assistance in improving documentation and code. *Please* clone the repository and support use by submitting pull requests fixing typos and clarifying documentation.


Installation
------------

If you aren't familiar at all with Python, please see  `Installing Python <https://github.com/vibrationtoolbox/vibration_toolbox/blob/master/docs/Installing_Python.rst>`_.

Installation is made easy with ``pip`` (or ``pip3``), with releases as we have time while we try
to create a full first release. Much of it works already, but we certainly need
issue reports (on `github <http://github.com/Vibration-Testing/vibrationtesting>`_).

To install (may not be available yet)::

  pip install --user vibrationtesting

where ``--user`` isn't necessary if you are using a locally installed version of Python such as `Anaconda <https://www.continuum.io/downloads>`_.

To run, I recommend you open a `Jupyter <https://jupyter.org>`_ notebook by using ``jupyter notebook`` and then type::

  import vibration_toolbox as vtb

For examples, see the `JupyterNotebooks folder <https://github.com/Vibration-Testing/vibrationtesting/tree/master/JupyterNotebooks>`_. Some of these have interactive capabilities that are only apparent when you run them yourself instead of just looking at them on github. Unfortunately our organization of these still leaves a little to be desired.

Installation of current code
____________________________

The usage documentation is far behind the current code, while the reference is way ahead of the released code due to the `autodoc <http://www.sphinx-doc.org/en/stable/ext/autodoc.html>`_ capability of `Sphinx <http://www.sphinx-doc.org/en/stable/>`_. Especially as of early 2017, the code is in rapid development. So is the documentation. Releases to `pypi <https://pypi.python.org/pypi>`_ are far behind current status as stopping to deploy would cost more time that it is worth. We have the objective of releasing a first non-beta version at the end of May, but even this cannot be promised.

If you wish to install the current version of the software, download the newest ``wheel`` file from
``https://github.com/vibrationtoolbox/vibration_toolbox/tree/master/dist``. This should be easy to understand, outside of what a ``wheel file`` is. Don't fret that. It's the full package as it currently stands (at lease the last time someone made a `wheel`).

Then, type::

  pip install --force-reinstall --upgrade --user dist/vibrationtesting-0.5b9-py3-none-any.whl

Where you see ``0.5.b9`` above, you will likely need to insert the correct version. We don't increment versions for each minor edit, so you may have to force upgrade.

That should be it. Please note issues on the `issues tab <https://github.com/Vibration-Testing/vibrationtesting/issues>`_ on github.
