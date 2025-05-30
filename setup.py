from setuptools import find_packages, setup

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mlutils",
    version="0.1.0",
    packages=find_packages(include=["mlutils", "mlutils.*"]),
    install_requires=requirements,
    include_package_data=True,
    description="Utilities for machine learning projects.",
    author="Dan Ogawa Lillrank",
    url="https://github.com/yourusername/mlutils",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
