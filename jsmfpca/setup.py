import os
from setuptools import setup, find_packages

# Read the README.md for the long description if it exists in this directory
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="jsmfpca",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description=("Joint Spectral Multilevel Functional "
                 "Principal Component Analysis"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "scipy>=1.12.0",
        "scikit-learn>=1.5.2",
        "pandas>=2.2.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=("functional data analysis, fda, circadian rhythms, "
              "biosignal analysis, heart rate, pca"),
    python_requires=">=3.8",
)
