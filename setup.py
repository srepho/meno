"""Setup script for the meno package."""

from setuptools import setup, find_packages

setup(
    name="meno",
    version="1.0.1",
    description="Topic modeling toolkit for messy text data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Stephen Oates",
    author_email="stephen.oates@example.com",
    url="https://github.com/srepho/meno",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8,<3.14",
    install_requires=[
        "pandas>=2.0.0,<3.0.0",
        "pyarrow>=11.0.0",
        "scikit-learn>=1.2.0,<2.0.0",
        "pydantic>=2.0.0,<3.0.0",
        "pyyaml>=6.0,<7.0",
        "jinja2>=3.1.2,<4.0.0",
        "thefuzz>=0.20.0,<0.21.0",
        "argparse>=1.4.0,<2.0.0",
    ],
    extras_require={
        "minimal": [
            "sentence-transformers>=2.2.2,<3.0.0", 
            "transformers>=4.28.0,<5.0.0",
            "torch>=2.0.0,<3.0.0",
            "plotly>=5.14.0,<6.0.0",
            "umap-learn>=0.5.3,<0.6.0",
            "hdbscan>=0.8.29,<0.9.0", 
            "bertopic>=0.15.0,<0.16.0",
            "gensim>=4.3.0,<5.0.0",
            "spacy>=3.5.0,<4.0.0",
            "wordcloud>=1.9.0,<2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "meno-config=meno.cli.team_config_cli:main",
        ],
    },
    license="MIT",
)