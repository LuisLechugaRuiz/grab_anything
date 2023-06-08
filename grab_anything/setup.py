from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="grab_anything",
    version="0.1.0",
    author="Luis Lechuga Ruiz",
    author_email="luislechugaruiz@gmail.com",
    description="Grab anything, an initial version",
    packages=find_packages(),
    install_requires=requirements,
)
