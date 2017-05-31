#!/usr/bin/env python
# coding=utf-8


from setuptools import setup, find_packages


setup(
    name="decap",
    version="0.0.2",
    author="jkmiao",
    author_email="miao1202@126.com",
    description= "verify code of most types",
    url = "http://www.cnblogs.com/jkmiao",
    package_data = {
        '': ['*.txt', '*.model']
    },
    lisense = 'MIT',
    install_requires=[
        "sklearn",
        "numpy",
        "scipy",
        "pillow",
        "tornado",
        "pillow",
        "keras",
        "h5py",
        "tensorflow"
        ],
    packages=find_packages(),
)
