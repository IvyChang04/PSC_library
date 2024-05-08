import time
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import argparse
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from pathlib import Path
from ParametricSpectralClustering.psc import PSC

ROOT = Path("JSS_Experiments").parent.absolute()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-dataset",
    "--dataset",
    type=str,
    help="the dataset used in this single experiment",
)
parser.add_argument("-model_path", "--path", default=None, type=str, help="model path")
args = parser.parse_args()

r = 72
rng = np.random.RandomState(r)
torch.manual_seed(0)
random.seed(int(r))
np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 10000
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=rng
)
x, y = noisy_circles
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=rng)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(7.5, 3))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
)

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 10,
    "n_clusters": 3,
    "min_samples": 20,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

datasets = [
    (
        "noisy_circles",
        noisy_circles,
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 20,
            "xi": 0.25,
        },
    ),
    (
        "noisy_moons",
        noisy_moons,
        {"damping": 0.75, "preference": -220, "n_clusters": 2},
    ),
    (
        "varied",
        varied,
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 5,
            "xi": 0.035,
            "min_cluster_size": 0.2,
        },
    ),
    (
        "aniso",
        aniso,
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 20,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    ("blobs", blobs, {}),
    ("no_structure", no_structure, {}),
]

if args.dataset == "noisy_circles":
    datasets = [datasets[0]]
    fig_num = 1
elif args.dataset == "noisy_moons":
    datasets = [datasets[1]]
    fig_num = 2
elif args.dataset == "varied":
    datasets = [datasets[2]]
    fig_num = 3
elif args.dataset == "aniso":
    datasets = [datasets[3]]
    fig_num = 4
elif args.dataset == "blobs":
    datasets = [datasets[4]]
    fig_num = 5
elif args.dataset == "no_structure":
    datasets = [datasets[5]]
    fig_num = 6


# for noisy_circles, noisy_moons, blobs, no_structure
class Net1(nn.Module):
    def __init__(self, out_put):
        super(Net1, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, out_put)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.output_layer(x)
        return x


# for varied
class Net2(nn.Module):
    def __init__(self, out_put):
        super(Net2, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, out_put)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.output_layer(x)
        return x


# for aniso
class Net3(nn.Module):
    def __init__(self, out_put):
        super(Net3, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, out_put)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.output_layer(x)
        return x


for i_dataset, (name, dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    # print(name)
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    # KMeans = cluster.KMeans(
    #     n_clusters=params["n_clusters"],
    #     init="random",
    #     n_init="auto",
    #     algorithm="elkan",
    #     random_state=rng,
    # )
    # spectral = cluster.SpectralClustering(
    #     n_clusters=params["n_clusters"],
    #     eigen_solver="arpack",
    #     affinity="nearest_neighbors",
    #     random_state=rng,
    # )
    torch.manual_seed(0)
    model_1 = Net1(params["n_clusters"])
    torch.manual_seed(0)
    model_2 = Net2(params["n_clusters"])
    torch.manual_seed(0)
    model_3 = Net3(params["n_clusters"])
    kmeans_psc = cluster.KMeans(
        n_clusters=params["n_clusters"], random_state=rng, n_init=10, verbose=False
    )

    l = ["noisy_circles", "noisy_moons", "blobs", "no_structure"]
    if name in l:
        model = model_1
        # print("1")
    elif name == "varied":
        model = model_2
        # print("2")
    elif name == "aniso":
        model = model_3
        # print("3")

    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, Shape: {param.shape}")
    #     print(f"Parameter values:\n{param.data}\n")

    psc = PSC(
        model=model,
        clustering_method=kmeans_psc,
        sampling_ratio=0,
        n_components=params["n_clusters"],
        n_neighbor=params["n_neighbors"],
        batch_size_data=10000,
        random_state=rng,
    )

    clustering_algorithms = (
        # ("KMeans", KMeans),
        # ("SpectralClustering", spectral),
        ("PSC", psc),
    )

    for algo_name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category=UserWarning,
            )
            if args.path == None:
                algorithm.fit(X)
            else:
                filename = name + "_figure1.pkl"
                algorithm.load_model(ROOT / args.path / filename)
            # if algo_name == "PSC":
            #     for name, param in algorithm.model.named_parameters():
            #         print(f"Parameter name: {name}, Shape: {param.shape}")
            #         print(f"Parameter values:\n{param.data}\n")

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(np.int32)
        else:
            y_pred = algorithm.predict(X)

        if args.dataset == "noisy_circles":
            np.savetxt("y_pred_noisy_circles.txt", y_pred, fmt="%d", delimiter=",")
        elif args.dataset == "noisy_moons":
            np.savetxt("y_pred_noisy_moons.txt", y_pred, fmt="%d", delimiter=",")
        elif args.dataset == "varied":
            np.savetxt("y_pred_varied2.txt", y_pred, fmt="%d", delimiter=",")
        elif args.dataset == "aniso":
            np.savetxt("y_pred_aniso.txt", y_pred, fmt="%d", delimiter=",")
        elif args.dataset == "blobs":
            np.savetxt("y_pred_blobs.txt", y_pred, fmt="%d", delimiter=",")
        elif args.dataset == "no_structure":
            np.savetxt("y_pred_no_structure.txt", y_pred, fmt="%d", delimiter=",")

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(algo_name, size=10)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1
name = "Figure1-" + str(fig_num) + ".pdf"
fig_name = ROOT / "Synthesis_dataset" / name
plt.savefig(fig_name, format="pdf", bbox_inches="tight")
# f.close()
