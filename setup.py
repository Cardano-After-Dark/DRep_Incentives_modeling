from setuptools import setup, find_packages

setup(
    name="drepsim",
    version="0.1.0",
    description="DRep Incentivization Simulation Framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
        "typeguard>=2.13.0",
        "pytest>=7.0.0",
    ],
    entry_points={
        "console_scripts": [
            "drepsim=drepsim.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 