from setuptools import setup, find_packages

setup(
    name="flare-gas-recovery",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "numpy>=1.24",
    ],
    author="Ing. Kelvin Cabrera",
    description="ML-based flare gas recovery optimization system",
    python_requires=">=3.9",
)
