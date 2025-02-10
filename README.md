<!-- Parametric Spectral Clustering -->

# Parametric Spectral Clustering

This repository provides a PyTorch implementation of **Parametric Spectral Clustering (PSC)**, an advanced alternative to traditional spectral clustering. PSC addresses critical challenges in computational efficiency, memory consumption, and the lack of online learning capabilities. It serves as a scalable framework for applying spectral clustering to large datasets.

---

## Installation

### Dependencies

PSC requires the following dependencies:

- Python (>= 3.8)
- NumPy (>= 1.26.4)
- SciPy (>= 1.13.0)
- PyTorch (>= 2.2.2)
- scikit-learn (>= 1.4.2)
- Pandas (>= 2.2.2)
- Matplotlib (>= 3.8.4)

### User Installation

Use setup.py:

```sh
python setup.py install
```

Use pip:

```sh
pip install git+https://github.com/IvyChang04/PSC_library.git
```

## Sample Usage

### Example: Clustering Handwritten Digits Using Python Code

The following example demonstrates PSC applied to the **UCI ML handwritten digits dataset**.

```python
from ParametricSpectralClustering import PSC, Four_layer_FNN
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

# Load and normalize dataset
digits = load_digits()
X = digits.data / 16

# Define clustering method
cluster_method = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')

# Define PSC model
model = Four_layer_FNN(64, 128, 256, 64, 10)
psc = PSC(model=model, clustering_method=cluster_method, n_neighbor=10, sampling_ratio=0, batch_size_data=1797)

# Train the PSC model
psc.fit(X)

# Save and apply model
psc.save_model("model")
cluster_idx = psc.predict(X)
```

<!-- COMMEND LINE TOOL -->

### Example: Clustering Handwritten Digits Using Command Line Tool

After installation, you may run the following scripts directly.

```sh
python bin/run.py [data] [rate] [n_cluster] [model_path] [cluster_result_format]
```

**Arguments:**

* `[data]` - Path to the dataset (`.txt`, `.csv`, or `.npy` formats supported).
    
* `[rate]` - Proportion of data (between 0.0 and 1.0) reserved for training the mapping function from the original feature space to the spectral embedding.

* `[n_cluster]` - Number of clusters (must be less than the total dataset size).

* `[model_path]` - Path to save the trained model.

* `[cluster_result_format]` - Format of cluster results (`.txt` or `.csv`).

<!-- EXPERIMENT-->

# Experiment

The `JSS_Experiments` directory contains the experiments used in our study: "_PSC: A Python Package for Parametric Spectral Clustering_."

To run the experiments:

```sh
cd JSS_Experiments
python run_exp.py
```

The script:

1. Generates two synthetic datasets: "**Double Circles**" and "**Double Moons**".

1. Produces scatter plots for visualization.

1. Applies PSC to cluster these datasets.

1. Colors the scatter plot based on the assigned cluster IDs.

<!-- Test -->

# Test

To run unit tests, use:

```sh
pytest tests
```

<!-- LICENSE -->

# License

This project is licensed under the MIT License. See `LICENSE.txt` for details.

<!-- CONTACT -->

# Contact

For questions or collaborations, contact the authors:

* **Ivy Chang**: ivy900403@gmail.com

* **Hsin Ju Tai**: hsinjutai@gmail.com

* **Hung-Hsuan Chen**: hhchen1105@acm.org

