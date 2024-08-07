from setuptools import setup, find_packages
import os

setup(
    name="geom_cairo",
    version="0.0.1",
    author="Kyle Stewart",
    author_email="kylestew@gmail.com",
    packages=find_packages(),
    description="Cairo 2D rendering utils for geom library",
    long_description=open("README.md").read(),
    install_requires=[
        "pycairo",
        "numpy",
        "pillow",
        "geom @ file://localhost/%s/../geom/" % os.getcwd().replace("\\", "/"),
    ],
)
