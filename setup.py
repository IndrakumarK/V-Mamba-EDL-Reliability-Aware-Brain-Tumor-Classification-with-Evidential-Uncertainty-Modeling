from setuptools import setup, find_packages

setup(
    name="vmamba_edl",
    version="1.0.0",
    description="V-Mamba-EDL: Reliability-Aware Brain Tumor Classification with Evidential Uncertainty Modeling",
    author="Indrakumar K",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "pillow>=9.5.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
