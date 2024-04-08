from setuptools import setup, find_packages


with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='pystorm3',
   version='0.1.0',
   description='Python implementation of some Brainstorm functions',
   license="tbd",
   long_description=long_description,
   author='Dominic Boutet',
   author_email='dominic.boutet@mail.mcgill.ca',
   url="tbd",
   packages=['pystorm'],  #same as name
   install_requires=['torch==2.2.0','numpy>=1.26'], #external packages as dependencies
   python_requires='>=3.9',
   #scripts=[
   #        ]
)