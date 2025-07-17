#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.rst") as history_file:
#     history = history_file.read()

requirements = []

setup(
    author="Anowar Shajib",
    author_email="ajshajib@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    description="Software to clean modulation/wiggles due to resampling noise in the JWST/NIRSpec IFS spectra.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    name="raccoon",
    packages=find_packages(where=".", include=["raccoon", "raccoon.*"]),
    package_dir={"": "."},
    url="https://github.com/ajshajib/raccoon",
    version="1.0.0",
    zip_safe=False,
)
