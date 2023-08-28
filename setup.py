from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

requirements = [
    'torch >= 1.12.1',
    'numpy >= 1.19.2',
    'scikit-learn >= 1.1.2',
    'scipy >= 1.7.3',
    'pandas >= 1.5.3'
]

scripts = [
    'bin/run_psc.py'
]

setup (
    name='ParametricSpectralClustering',
    version='0.0.18',
    description='A library for users to use parametric spectral clustering',
    long_description=open("README.md").read() + '\n\n' + open("CHANGELOG.txt").read(),
    long_description_content_type = "text/markdown",
    packages=find_packages(exclude=["tests*"]),
    classifiers=classifiers,
    url='',
    author='Ivy Chang, Hsin Ju Tai',
    author_email='ivy900403@gmail.com, luludai020127@gmail.com',
    license='MIT',
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    python_requires = ">=3.8",
    scripts=scripts
)
