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
    clustering_method=cluster.KMeans(n_clusters=2, n_init=10, verbose=False),
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
