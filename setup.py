from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("src/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dash-nyu",
    version="1.0.0",
    author="Your Name",
    author_email="your-email@domain.com",
    description="Domain-Aware Sparse Hypernetworks with Phoenix neural ODE and ensemble pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dash-nyu",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "dash-train=src.train_DASH_simu:main",
            "dash-ensemble=src.ensemble_dash:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.cfg", "*.txt", "*.md"],
    },
    keywords="neural-ode, gene-regulatory-networks, bioinformatics, machine-learning, ensemble-methods",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/dash-nyu/issues",
        "Source": "https://github.com/yourusername/dash-nyu",
        "Documentation": "https://github.com/yourusername/dash-nyu#readme",
    },
)
