Reporting bugs
--------------
If you find a bug, please open an issue on the `Github issues tracker <https://github.com/Vibration-Testing/vibrationtesting/issues>`_.
Please provide some code that reproduces the error and versions of the packages installed.

Contributing code
-----------------
To contribute code we recommend you follow these steps:

To contribute code we recommend you follow these steps:

#. Fork the repository on github

#. Set up travis-ci for your branch. This is actually pretty quick and easy:

  #. Go to travis-ci.org and Sign in with GitHub.

  #. Account page will show all the repositories attached to GitHub.

  #. Find the right repository and enable Travis CI.

  #. Once this is done, Travis CI will be turned-on in GitHub fork.

  #. Go back to the fork on GitHub, click    

     ``Settings`` -> ``Webhooks`` -> ``updated travis-ci.org link``.

  #. Default or customize the options based on needs and click 

     ``Update webhook``. 	  

#. Clone the repository to your favorite location on your drive where you want to work on it.

#. To work in `developer mode <https://packaging.python.org/distributing/#working-in-development-mode>`_, from a terminal (python enabled) at the top level directory inside the ``vibration testing module`` type::

    $ pip install -e .

   This will allow you to edit the code while having it pretend to be installed. Keep in mind, if you have actually installed the ``vibration testing module`` you may have a conflict. You must uninstall it and install your development version with the command above.

#. If a new function is added
   please provide docstrings following the `Numpy standards for docstrings <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.
   The docstrings should contain examples to be tested.

   Specifically note:

   1. Parameters should be listed similarly to:

    |    filename : str
    |    copy : bool
    |    dtype : data-type
    |    iterable : iterable object
    |    shape : int or tuple of int
    |    files : list of str
    |    time : array_like

   2. First line should be inline with the ``"""`` and brief enough to fit on one line.

   3. There must be a blank line after the first line.

   This is not exhaustive. It just highlights some consistent errors made.

#. Run the doctests regularly when you make edits.

   To run the doctests `<pytest https://docs.pytest.org/en/latest/>`_ is needed and can be installed with ``pip install -U pytest``.

   To run the tests from the shell you can `cd` to the project's root directory and type::

     $ pytest


    1. To run the tests from ``pycharm`` you can do: Run -> Edit Configurations -> Add -> python tests -> pytest Then just set the path to the project directory.

    2. To run the tests from ``spyder`` see `spyder-unittest <https://github.com/spyder-ide/spyder-unittest`_.

#. Commit and check `travis-ci <https://travis-ci.org/Vibration-Testing/vibrationtesting>`_ tests regularly. Having a great number of changes before a commit can make tracing errors very hard. Make sure you are looking at your branch when assessing whether it's working.

#. You may need to `update from the main repository <https://www.sitepoint.com/quick-tip-sync-your-fork-with-the-original-without-the-cli/>`_ before submitting a pull request. This allows you to see the complete results before we look at them.  If it doesn't work, the pull will (should) be denied. This can be a bit daunting for some, so it's recommended but not necessary.

#. If the tests are passing, make a git pull (in your GitHub app) to assure that your code is up to date with the master branch and that your code has no conflicts with the current base. Doing this regularly ensures that your accumulated edits won't be massively in conflict with the existing code base. After that, push your branch to github and then open a pull request.

#. Please provide feedback and corrections to these instructions.

Instructions bellow are directed to main developers
===================================================

To make distribution and release
--------------------------------

1) Edit the version number in ``vibrationtesting/__init__.py``
2) Use the Makefile, ``make release``

The ``conf.py`` file for the documentation pulls the version from ``__init__.py``

To make a distribution (for testing or posting to github)
-----------------------------------------------------------

.. code-block:: bash

  >> make wheel

To test before release
----------------------

TravisCI does this for us. 

See `notes <https://packaging.python.org/distributing/#working-in-development-mode>`_ on working in development mode.

To test distribution installabilty
-----------------------------------
Note: these are out of date.

python setup.py register -r pypitest
python setup.py sdist upload -r pypitest

look at https://testpypi.python.org/pypi

Other information sites
------------------------

`twine notes <https://packaging.python.org/distributing/#working-in-development-mode>`_

https://pypi.python.org/pypi/wheel
