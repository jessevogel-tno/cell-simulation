from setuptools import find_packages, setup

setup(
    name="risq",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
