<!-- Parametric Spectral Clustering -->
## Parametric Spectral Clustering

TODO

<!-- PREREQUISITES -->
<!-- ## Prerequisites

Things you need to install before started.
* torch
    ```sh
    pip3 install torch torchvision torchaudio
    ```
* numpy
    ```sh
    pip install numpy
    ```
* sklearn
    ```sh
    pip3 install -U scikit-learn
    ```
* scipy
    ```sh
    python -m pip install scipy
    ```
* pickle
    ```sh
    pip3 install pickle5
    ``` -->

<!-- SAMPLE USAGE -->
## Sample Usage

Using UCI ML hand-written digits datasets as an example.

```sh
>>> from PSC_lib import PSC, Net
>>> from sklearn.datasets import load_digits
>>> from sklearn.cluster import KMeans
>>> digits = load_digits()
>>> X = digits.data/16
>>> cluster_method = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
>>> model = Net(64, 128, 256, 64, 10)
>>> psc = PSC(model=model, clustering_method=cluster_method)
>>> cluster_idx = psc.fit_predict(X)
```

<!-- INSTALLATION -->
## Installation
```sh
python setup.py install
```
<!-- COMMEND LINE TOOL -->
## Commend line tool

TODO

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contact

Your Name - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)