from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.17'
DESCRIPTION = 'Speed Bootstrap'
LONG_DESCRIPTION = 'Boostrap fast using parallel processing for vector-valued statistics. Plot estimates and compute various confidence intervals. To see a quick demo click <a href="https://github.com/fcgrolleau/speedboot/blob/main/speedboot/demo.ipynb">here</a>. Source code is available <a href="https://github.com/fcgrolleau/speedboot/blob/main/speedboot/speedboot.py">there</a>.'

# Setting up
setup(
    name="speedboot",
    version=VERSION,
    author="François Grolleau",
    author_email="<francois.grolleau@aphp.fr>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['joblib', 'tqdm'],
    keywords=['bootstrap', 'confidence intervals', 'statistics', 'inference', 'parallel'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)