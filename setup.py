from setuptools import setup, find_packages

setup(
    name='caramel',
    version='0.1.0',
    scripts=['bin/construct.py'],
    description='An package for constructing compressed static functions (CSFs)',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[
    "bitarray==2.6.0",
    "spookyhash==2.1.0",
    "pytest",
],
)
