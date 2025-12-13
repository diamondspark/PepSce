from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pepsce",
    version="0.1.0",
    author="M. Pandey",
    description="Scalable Reinforcement Learning for Bioactive peptide discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diamondspark/PepSce",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "stable-baselines3>=1.5.0",
        "pyyaml>=5.4",
        "easydict>=1.9",
        "tqdm>=4.62.0",
        "fair-esm>=2.0.0",
        "ott-jax>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
        "logging": [
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tarsa-train=screening.sample_mu:main",
        ],
    },
)