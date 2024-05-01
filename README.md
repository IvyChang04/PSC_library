<!-- Parametric Spectral Clustering -->

# Parametric Spectral Clustering

This repository provides a PyTorch implementation of the **Parametric Spectral Clustering** (PSC) algorithm, which offers a favorable alternative to the traditional spectral clustering algorithm. PSC addresses issues related to computational efficiency, memory usage, and the absence of online learning capabilities. It serves as a versatile framework suitable for applying spectral clustering to large datasets.

<!-- PREREQUISITES -->

# Installation

## Dependencies

Parametric Spectral Clustering requires:

-   Python (>= 3.8)
-   NumPy (>= 1.26.4)
-   SciPy (>= 1.13.0)
-   PyTorch (>= 2.2.2)
-   scikit-learn (>= 1.4.2)
-   Pandas (>= 2.2.2)
-   Matplotlib (3.8.4)

---

<!-- INSTALLATION -->

## User installation

Use setup.py:

```sh
python setup.py install
```

Use pip:

```sh
pip install ParametricSpectralClustering
```

<!-- SAMPLE USAGE -->

## Sample Usage

Using UCI ML hand-written digits datasets as an example.

```sh
>>> from ParametricSpectralClustering import PSC, Four_layer_FNN
>>> from sklearn.datasets import load_digits
>>> from sklearn.cluster import KMeans
>>> digits = load_digits()
>>> X = digits.data/16
>>> cluster_method = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
>>> model = Four_layer_FNN(64, 128, 256, 64, 10)
>>> psc = PSC(model=model, clustering_method=cluster_method, n_neighbor=10, sampling_ratio=0, batch_size_data=1797)
>>> psc.fit(X)
>>> psc.save_model("model")
>>> cluster_idx = psc.predict(X)
```

<!-- COMMEND LINE TOOL -->

## Command line tool

After installation, you may run the following scripts directly.

```sh
python bin/run.py [data] [rate] [n_cluster] [model_path] [cluster_result_format]
```

The `[data]` can accept .txt, .csv, and .npy format of data.

The `[rate]` should be in float, between 0.0 and 1.0. It represent the proportion of the input data reserved for training the mapping function from the original feature space to the spectral embedding.

The `[n_cluster]` is the number of clusters the user intends to partition. This number needs to be lower than the total data points available within the dataset.

The `[model_path]` is the path to save the trained model.

The `[cluster_result_format]` can be either .txt or .csv. It represent the format of the cluster result.

<!-- EXPERIMENT-->

# Experiment

The 'JSS_Experiments' directory contains the code for the experiments detailed in the paper "PSC: a Python Package for Parametric Spectral Clustering." This includes scripts for experiments on the Firewall, NIDS, and Synthesis datasets.

Prior to executing these scripts, ensure that the necessary datasets have been downloaded and placed in the appropriate location. The datasets can be obtained from the following sources:

-   NIDS Dataset: https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset

Please place the downloaded datasets in the ‘JSS_Experiments/datasets’ directory. Ensure the datasets are correctly located before running the scripts.

```sh
cd JSS_Experiments
python run.py
```

<!-- Test -->

# Test

To run the test, use the following command:

```sh
pytest tests
```

<!-- LICENSE -->

# License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- CONTACT -->

# Contact

| Author | Ivy Chang           | Hsin Ju Tai         |
| ------ | ------------------- | ------------------- |
| E-mail | ivy900403@gmail.com | hsinjutai@gmail.com |

Project Link: [Parametric Spectral Clsutering](https://github.com/IvyChang04/PSC_library)
