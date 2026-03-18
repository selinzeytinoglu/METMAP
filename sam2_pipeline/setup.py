from setuptools import find_packages, setup

setup(
    name="sam2",
    version="1.0",
    packages=find_packages(),
    package_data={
        "sam2": ["configs/**/*.yaml"],
    },
)
