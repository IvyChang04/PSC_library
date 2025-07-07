# Parametric Spectral Clustering

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/IvyChang04/PSC_library/blob/main/LICENSE.txt)
[![PyPI version](https://badge.fury.io/py/psc-clustering.svg)](https://badge.fury.io/py/psc-clustering)
[![Tests](https://github.com/IvyChang04/PSC_library/actions/workflows/main.yml/badge.svg)](https://github.com/IvyChang04/PSC_library/actions)

Parametric Spectral Clustering (PSC) is a high-performance PyTorch library that makes spectral clustering feasible for big data and streaming applications. It addresses the critical challenges of traditional spectral clustering, including computational efficiency, memory consumption, and the lack of incremental learning capabilities.

---

## ✨ Key Features

* 🚀 **Scalable for Big Data**: Efficiently handles large datasets by learning a parametric mapping, avoiding the need to build a full affinity matrix.
* 💡 **Incremental Learning**: Easily clusters new, unseen data points using the trained model without retraining from scratch.
* 🧠 **Flexible Model Design**: Integrates with any custom `torch.nn.Module`, giving you full control over the mapping function's architecture.
* 🐍 **Scikit-learn Compatible API**: Uses the familiar `.fit()` and `.predict()` interface for seamless integration into existing ML workflows.

## 📦 Installation

You can install the package via pip from PyPI:
```sh
pip install psc-clustering
```

Alternatively, you can install the latest version directly from GitHub:
```bash
pip install git+[https://github.com/IvyChang04/PSC_library.git](https://github.com/IvyChang04/PSC_library.git)
```

### Dependencies:
PSC requires Python >= 3.8 and the following core libraries:

* PyTorch
* scikit-learn
* NumPy
* SciPy
* Pandas (optional, for data loading)
* Matplotlib (optional, for examples running)

## 🚀 Quick Start

Here is a simple example of clustering the "moons" dataset. The library provides default models, so you can get started in just a few lines of code.

```python
import matplotlib.pyplot as plt
from sklearn import datasets
from ParametricSpectralClustering import PSC, Four_layer_FNN

# 1. Generate data
X, y = datasets.make_moons(n_samples=1000, noise=0.05, random_state=42)

# 2. Define a model and initialize PSC
# An input dimension of 2 (for the data) and output of 2 (for n_clusters)
model = Four_layer_FNN(2, 16, 32, 16, 2) 
psc = PSC(n_clusters=2, model=model, random_state=42)

# 3. Fit the model and predict clusters
y_pred = psc.fit(X).predict(X)

# 4. Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="rainbow")
plt.title("PSC Clustering Result")
plt.axis("equal")
plt.show()
```

```diff
- show the figure
```

## 📚 Detailed Examples 
<details>
  <summary>Click to see more examples (Double Circles, Digits)</summary>
  **Example: Clustering Double Circles Dataset**

  The double circles dataset is challenging because one cluster is inside another. PSC learns to map these points to a space where they are linearly separable.

  ```python
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from ParametricSpectralClustering.psc import PSC

# learn the mapping from feature space to spectral space
class Net1(nn.Module):
    def __init__(self, out_put):
        super(Net1, self).__init__()
        self.fc = nn.Linear(2, 32)
        self.output_layer = nn.Linear(32, out_put)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

n_samples = 1000
X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

psc = PSC(
    model=Net1(2),
    n_clusters=2,
    sampling_ratio=0,
    n_components=2,
    n_neighbor=10,
    batch_size_data=len(X)
)
psc.fit(X)
y_pred = psc.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="rainbow")
plt.axis("equal")
plt.show()
plt.close()
  ```

Here is the clustering result.
```diff
- show the figure.
```

**Example: Clustering Handwritten Digits Using Python Code**

The following example demonstrates PSC applied to the UCI ML handwritten digits dataset.

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

</details>

## 🛠️ Command Line Tool

After installation, you can run clustering directly from the command line.

**Syntax:**

```bash
python bin/run.py [data] [rate] [n_cluster] [model_path] [cluster_result_format]
```

**Arguments:**

* `[data]` - Path to the dataset (`.txt`, `.csv`, or `.npy`).
* `[rate]` - Proportion of data for training the mapping function (0.0 to 1.0).
* `[n_cluster]` - Number of clusters.
* `[model_path]` - Path to save the trained model.
* `[cluster_result_format]` - Output format for cluster results (`.txt` or `.csv`).

**Example:**

```bash
python bin/run.py data/digits.csv 0.2 10 models/digits_psc.pt results/clusters.csv
```

## 📝 API Reference

<details>
<summary>Click to see all PSC Class Parameters and Guidelines</summary>
The `PSC` class is the main interface for parametric spectral clustering.

**Core Parameters**
|Parameter          |Type           |Default                         |Description                                                             |
|-------------------|---------------|--------------------------------|------------------------------------------------------------------------|
|`n_clusters`       |int            |10                              |Number of clusters to find in the data                                  |
|`n_components`     |int            |0                               |Number of embedding dimensions. If 0, defaults to n_clusters            |
|`n_neighbor`       |int            |8                               |Number of neighbors for k-nearest neighbors graph construction          |
|`model`            |torch.nn.Module|Four_layer_FNN(64,128,256,64,10)|Neural network to learn the mapping from feature space to spectral space|
|`clustering_method`|sklearn.cluster|KMeans                          |Clustering algorithm to apply to the learned embeddings                 |

**Training Parameters**
|Parameter              |Type |Default|Description                                      |
|-----------------------|-----|-------|-------------------------------------------------|
|`epochs`               |int  |50     |Number of training epochs for the neural network |
|`sampling_ratio`       |float|0.3    |Proportion of data used for training (0.0 to 1.0)|
|`batch_size_data`      |int  |50     |Batch size for processing data chunks            |
|`batch_size_dataloader`|int  |20     |Batch size for neural network training           |
|`random_state`         |int  |None   |Random seed for reproducibility                  |

**Advanced Parameters**
|Parameter              |Type |Default|Description                                      |
|-----------------------|-----|-------|-------------------------------------------------|
|`criterion`            |torch.nn.modules.loss|nn.MSELoss()|Loss function for training the neural network    |
</details>

## ✅ Code Testing

To run unit tests, clone the repository and run pytest:

```bash
git clone [https://github.com/IvyChang04/PSC_library.git](https://github.com/IvyChang04/PSC_library.git)
cd PSC_library
pytest tests
```

## 📄 Citation

```bibtex
@inproceedings{chen2023toward,
  title={Toward Efficient and Incremental Spectral Clustering via Parametric Spectral Clustering},
  author={Chen, Jo-Chun and Chen, Hung-Hsuan},
  booktitle={2023 IEEE International Conference on Big Data (BigData)},
  pages={1070--1075},
  year={2023},
  organization={IEEE}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to fork the repository, make changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

For questions or collaborations, contact the authors:

* Ivy Chang: ivy900403@gmail.com
* Hsin Ju Tai: hsinjutai@gmail.com
* Hung-Hsuan Chen: hhchen1105@acm.org