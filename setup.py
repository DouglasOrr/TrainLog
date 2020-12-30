"""Package setup."""

import pathlib

import setuptools  # type: ignore

import trainlog._version

project_dir = pathlib.Path(__file__).parent

setuptools.setup(
    name="trainlog",
    version=trainlog._version.__version__,  # pylint:disable=protected-access
    description="A simple logging library, designed for deep learning",
    long_description=(project_dir / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/DouglasOrr/TrainLog",
    author="Douglas Orr",
    license="MIT",
    packages=["trainlog"],
    install_requires=[
        req for req in (project_dir / "requirements.txt").read_text().split("\n") if req
    ],
)
