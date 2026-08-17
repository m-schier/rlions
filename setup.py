from setuptools import setup, find_packages

setup(
    name="rlions",
    version="0.1",
    author="Maximilian Schier, Lea Richtmann",
    description=(
        "Ion shuttling compiler for trapped-ion quantum computers using reinforcement learning"
    ),
    license="AGPL-3.0-or-later",
    packages=find_packages(),
    install_requires=[
        'chex>=0.1.86',
        'flax>=0.8.4',
        'jax~=0.4.28',
        'mlflow>=3.8.1',
        'optax>=0.2.2',
    ],
    extras_require={
        'test': ['pytest>=8.2.2'],
    },
)
