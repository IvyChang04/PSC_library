<!-- Parametric Spectral Clustering -->
# Parametric Spectral Clustering

This repository provides a PyTorch implementation of the **Parametric Spectral Clustering** (PSC) algorithm, which offers a favorable alternative to the traditional spectral clustering algorithm. PSC addresses issues related to computational efficiency, memory usage, and the absence of online learning capabilities. It serves as a versatile framework suitable for applying spectral clustering to large datasets.

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
After installation, you may run the following scripts directly.

```sh
python PSC_lib.py [train_data] [n_cluster] [test_splitting_rate]
```

The ``[train_data]`` can accept .txt, .csv, and .npy format of data.

The ``[n_cluster]`` is the number of clusters to form as well as the number of centroids to generate.

The ``[test_splitting_rate]`` should be in float, between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contact
|Author|Ivy Chang|Hsin Ju Tai|
|---|---|---|
|E-mail|ivy900403@gmail.com|luludai020127@gmail.com|

Project Link: [TBA](https://github.com/your_username/repo_name)
