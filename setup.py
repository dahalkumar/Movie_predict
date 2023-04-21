"""
author:@kumar dahal
this function is written for version controlling.
It will read long description from readme.md file
"""
#import setuptools
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
#set the initial version as 0
__version__ = "0.0.0"

REPO_NAME = "movie_predictor"
AUTHOR_USER_NAME = "kumar dahal"
SRC_REPO = "movie_prediction"
AUTHOR_EMAIL = "kumardahal536@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for Regression problem",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    #for now we have only one package directory so we have given src, also it is our main src directory
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)