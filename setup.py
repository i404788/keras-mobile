from setuptools import setup
from setuptools import find_packages
import os


setup(name='keras_mobile',
      version='1.0.2',
      description='Fast & Compact keras blocks and layers for use in mobile applications',
      author='Ferris Kwaijtaal',
      author_email='ferrispc@hotmail.com',
      url='https://github.com/i404788/keras-mobile',
      license='MIT',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
