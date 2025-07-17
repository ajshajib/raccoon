==========
🦝 raccoon
==========

|Read the Docs| |GitHub| |Codecov| |Black| |docformatter| |docstyle|


Raccoon cleans the "wiggles" (i.e., low-frequency sinusoidal artifacts) in the JWST-NIRSpec IFS (integral field spectroscopy) data. These wiggles are caused by resampling noise or aliasing artifacts. For a quick start or demonstration, see the `example notebook here`_.

.. _`example notebook here`: https://github.com/ajshajib/raccoon/blob/main/example/example_notebook.ipynb


Installation
------------

.. image:: https://img.shields.io/pypi/v/space-raccoon.svg
   :alt: PyPI - Version
   :target: https://pypi.org/project/space-raccoon/


You can install ``raccoon`` using ``pip``. Run the following command:

.. code-block:: bash

    pip install space-raccoon

Alternatively, you can install the latest development version from GitHub as:

.. code-block:: bash

    git clone https://github.com/ajshajib/raccoon.git
    cd raccoon
    pip install .

Features
--------

- **Wiggle cleaning**: The primary feature of the package is to clean wiggles from JWST-NIRSpec IFS data.
- **Visualization**: It provides tools to visualize the data before and after wiggle cleaning.
- **Documentation**: Comprehensive documentation is available to help users understand how to use the package effectively.
- |codecov| **tested**: The package includes unit tests to ensure the functionality works as expected.
- **User-Friendly**: The package aims to provide a user-friendly experience, with intuitive interfaces and clear documentation.

.. |Read the Docs| image:: https://readthedocs.org/projects/raccoon-docs/badge/?version=latest
    :target: https://raccoon-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |GitHub| image:: https://github.com/ajshajib/raccoon/actions/workflows/ci.yaml/badge.svg?branch=main
    :target: https://github.com/ajshajib/dolphin/actions/workflows/ci.yaml
    :alt: Build Status

.. |Codecov| image:: https://codecov.io/github/ajshajib/raccoon/graph/badge.svg?token=IZOMFPHA7W 
    :target: https://codecov.io/github/ajshajib/raccoon
    :alt: Code coverage

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |docstyle| image:: https://img.shields.io/badge/%20style-sphinx-0a507a.svg
    :target: https://www.sphinx-doc.org/en/master/usage/index.html

.. |docformatter| image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg
    :target: https://github.com/PyCQA/docformatter
