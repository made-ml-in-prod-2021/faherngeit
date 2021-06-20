from setuptools import find_packages, setup

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="First HW under MADE ml in prod course",
    author="Valeriy Tashchilin",
    install_requires=[
        "pandas>=1.2.4",
        "numpy>=1.18.0",
        "scikit-learn>=0.24.2",
        "PyYAML>=5.4.1",
        "pytest>=6.2.4",
    ],
    license="MIT",
)