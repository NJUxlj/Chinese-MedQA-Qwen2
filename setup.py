import os
from pathlib import Path

from setuptools import find_packages, setup

__version__=0.1

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "numpy<2.0.0",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "torchdata",
    "tensordict<=0.6.2",
    "transformers",
    "wandb",
    "packaging>=20.0",
]

TEST_REQUIRES = ["pytest", "pre-commit", "py-spy"]
MATH_REQUIRES = ["math-verify"]  # Add math-verify as an optional dependency
VLLM_REQUIRES = ["tensordict", "vllm"]

extras_require = {
    "test": TEST_REQUIRES,
    "math": MATH_REQUIRES,
    "vllm": VLLM_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="cmedqwen",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="",
    license="Apache 2.0",
    author="NJUxlj",
    author_email="",
    description="",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "cmedqwen": ["docs/*.md"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
