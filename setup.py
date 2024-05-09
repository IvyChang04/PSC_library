from setuptools import setup, find_packages

classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Operating System :: OS Independent",
]

requirements = [
    "scipy>=1.13.0",
    "torch>=2.2.2",
    "torchaudio>=2.2.2",
    "torchvision>=0.17.2",
    "scikit-learn>=1.4.2",
    "pandas>=2.2.2",
    "numpy==1.26.4",
    "matplotlib>=3.8.4",
]

scripts = [
    "bin/clustering.py",
    "bin/train_psc.py",
    "bin/run.py",
]

keywords = [
    "Spectral Clustering",
    "Incremental Clustering",
    "Online Clustering",
    "Non-linear clustering",
]

setup(
    name="ParametricSpectralClustering",
    version="0.0.4",
    description="A library for users to use parametric spectral clustering",
    long_description=open("README.md").read() + "\n\n" + open("CHANGELOG.txt").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    classifiers=classifiers,
    keywords=keywords,
    url="",
    author="Ivy Chang, Hsin Ju Tai",
    author_email="ivy900403@gmail.com, luludai020127@gmail.com",
    license="MIT",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
    scripts=scripts,
)
