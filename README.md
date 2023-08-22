<!-- Parametric Spectral Clustering -->
# Parametric Spectral Clustering

This repository provides a PyTorch implementation of the **Parametric Spectral Clustering** (PSC) algorithm. PSC is a novel spectral clustering algorithm that overcomes the limitations of the traditional spectral clustering algorithm at computational limitations, memory requirements, and inability to perform online learning. PSC is a general framework that can be applied to any spectral clustering algorithm. In this repository, we provide an example of PSC with the K-means clustering algorithm.

<!-- PREREQUISITES -->
# Installation
## Dependencies
Parametric Spectral Clustering requires:

* Python (>= 3.8)
* PyTorch (>= 1.19.2)
* Scikit-learn (>= 1.1.2)
* SciPy (>= 1.7.3)

---

<!-- INSTALLATION -->
## User installation

Use setup.py:
```sh
python setup.py install
```

Use pip:
```sh
pip install -i https://test.pypi.org/simple/ ParametricSpectralClustering==0.0.14
```

<!-- SAMPLE USAGE -->
## Sample Usage

Using UCI ML hand-written digits datasets as an example.

```sh
>>> from ParametricSpectralClustering import PSC, Net
>>> from sklearn.datasets import load_digits
>>> from sklearn.cluster import KMeans
>>> digits = load_digits()
>>> X = digits.data/16
>>> cluster_method = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
>>> model = Net(64, 128, 256, 64, 10)
>>> psc = PSC(model=model, clustering_method=cluster_method)
>>> cluster_idx = psc.fit_predict(X)
```

<!-- COMMEND LINE TOOL -->
## Command line tool

TODO

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contact
|Author|Ivy Chang|Hsin Ju Tai|
|---|---|---|
|E-mail|ivy900403@gmail.com|luludai020127@gmail.com|

Project Link: [TBA](https://github.com/your_username/repo_name)