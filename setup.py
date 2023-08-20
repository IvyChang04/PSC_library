from setuptools import setup

setup (
    name='PSC',
    version='0.1',
    description='Parametric Spectral Clustering',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    url='',
    author='',
    author_email='',
    license='',
    packages=['PSC'],
    install_requires=[
        'tourch',
        'numpy',
        'scikit-learn',
        'scipy',
        'pickle'
    ],
    include_package_data=True,
    zip_safe=False,
    scripts=['bin/PSC_library.py'],
)
