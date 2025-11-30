"""
Setup file for the Enhanced Stateful Router package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stateful-router",
    version="0.1.0",
    author="Pu Suo",
    description="A hierarchical neuro-symbolic architecture for verifiable chain of thought",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stateful-router",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "training", "training.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.12.0",
        "anthropic>=0.18.0",
        "pydantic>=2.5.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.2",
        "rich>=13.7.0",
        "tenacity>=8.2.3",
        "cachetools>=5.3.0",
        "jsonschema>=4.19.0",
        "tiktoken>=0.5.1",
        "sentence-transformers>=2.2.2",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "jsonlines>=4.0.0",
        "aiohttp>=3.9.0",
        "httpx>=0.25.0",
        "pyyaml>=6.0.1",
        "python-json-logger>=2.0.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.1",
        ],
        "training": [
            "transformers>=4.36.0",
            "torch>=2.1.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stateful-router=stateful_router.cli:main",
        ],
    },
)
