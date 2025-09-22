from setuptools import setup, find_packages
import pathlib

setup(
    name="SpecInfer",                # PyPI 上的名字，全局唯一
    version="1.0.0",    # 单源版本管理
    description="A simple framework implementing naive speculative decoding",
    author="Liwei Lan",
    author_email="llw25@mails.tsinghua.edu.cnsion",
    classifiers=[                    # 在 https://pypi.org/classifiers/ 挑选
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="sample, setuptools, development",
    packages=find_packages(exclude=["test_scripts*", "*.yaml", "mypy_cache*", ".vcode*"]),
    python_requires=">=3.8",

)