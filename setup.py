# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Setup of sam2act
Author: Haoquan Fang
"""
from setuptools import setup, find_packages

requirements = [
    "numpy",
    "scipy",
    "einops",
    "pyrender",
    "transformers",
    "omegaconf",
    "natsort",
    "cffi",
    "pandas",
    "tensorflow",
    "pyquaternion",
    "matplotlib",
    "bitsandbytes==0.38.1",
    #"triton==2.3.1",
    "transforms3d",
    "clip @ git+https://github.com/openai/CLIP.git",
    "wandb",
    "iopath",
    "hydra-core==1.3.2",
    "pip==23.0"
]

__version__ = "0.0.1"
setup(
    name="sam2act",
    version=__version__,
    description="sam2act",
    long_description="",
    author="Haoquan Fang",
    author_email="hqfang@cs.washington.edu",
    url="",
    keywords="robotics,computer vision",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=['sam2act'],
    install_requires=requirements,
    extras_require={
        "xformers": [
            "xformers @ git+https://github.com/facebookresearch/xformers.git@main#egg=xformers",
        ],
        "real": 
        [
            "robits[real,audio]>=0.5.1"
        ]
    },

    entry_points={
        'console_scripts': [
            'sam2act-agent = sam2act.real.sam2act_cli:cli',
        ]
    }
)
