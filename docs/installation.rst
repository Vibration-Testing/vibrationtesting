Installation
------------

.. toctree::
   :maxdepth: 3

   Installing_Python

Easy Installation
_________________

If you aren't familiar at all with Python, please see  `Installing Python <https://github.com/vibrationtoolbox/vibration_toolbox/blob/master/docs/Installing_Python.rst>`_.

Installation is made easy with ``pip`` (or ``pip3``), with releases as we have time while we try
to create a full first release. Much of it works already, but we certainly need
issue reports (on `github <https://github.com/Vibration-Testing/vibrationtesting/issues>`_).

To install::

  pip install --user vibrationtesting

where ``--user`` isn't necessary if you are using a locally installed version of Python such as `Anaconda <https://www.continuum.io/downloads>`_.

To run, I recommend you open a `Jupyter`_ notebook by using ``jupyter notebook`` and then type::

  import vibrationtesting as vt

For examples, see the `example ipynb notebooks <https://github.com/Vibration-Testing/vibrationtesting/tree/master/JupyterNotebooks>`_. Some of these have interactive capabilities that are only apparent when you load them with `Jupyter`_ instead of just looking at them on github. Sorry- these are very rough right now. You can see the `less rough notebooks <https://github.com/Vibration-Testing/vibrationtesting/tree/master/docs/tutorial/Notebooks>`_ used to make the manual as well.

Installation of current development version
___________________________________________

The usage documentation is far behind the current code, while the reference is way ahead of the released code due to the `autodoc <http://www.sphinx-doc.org/en/stable/ext/autodoc.html>`_ capability of `Sphinx <http://www.sphinx-doc.org/en/stable/>`_. Especially as of  2017, the code is in rapid development. So is the documentation. To get the most current version and stay up to date see the file `CONTRIBUTING.rst <https://github.com/Vibration-Testing/vibrationtesting/blob/master/CONTRIBUTING.rst>`_ in the github repository.

.. _Jupyter: jupyter.org
