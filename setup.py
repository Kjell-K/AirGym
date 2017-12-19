from setuptools import setup
import os

setup(
    name = "gym_airsim",
    version = "4.0",
    author = "Kjell Kersandt",
    author_email = "kjell.kersandt94@gmail.com",
    description = ("Integration of AirSim as OpenAI Gym enviroment for reinforcement learning"),
    license = "MIT",
    keywords = "reinforcement learning, AirSim, OpenAI GYM, Gym",
    url = "http://packages.python.org/an_example_pypi_project",
    install_requires=['keras=2.0.6'],
    extras_require={
          'gym': ['gym'],
      }
)
    