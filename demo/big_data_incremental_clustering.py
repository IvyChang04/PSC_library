import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn as nn
from sklearn import cluster
from sklearn.datasets import make_blobs
from ParametricSpectralClustering import PSC

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

# Generate a large dataset
X_train, y_train = make_blobs(n_samples=100000, centers=3, n_features=2, 
                              random_state=42, cluster_std=1.5)

# Train the PSC model
psc = PSC(
    model=Net1(3),
    clustering_method=cluster.KMeans(n_clusters=3, n_init=10, verbose=False),
    n_components=3,
    random_state=123,
    sampling_ratio=0.01)
start_time = time.time()
print(X_train.shape)
psc.fit(X_train)
end_time = time.time()
print(f"PSC training time: {end_time - start_time:.2f} seconds")

# Generate new (incremental)data
X_new, y_new = make_blobs(n_samples=1000, centers=3, n_features=2, 
                          random_state=42, cluster_std=1.5)

time_start = time.time()
new_labels = psc.predict(X_new)
end_time = time.time()
print(f"PSC incremental clustering (predicting) time: {end_time - time_start:.2f} seconds")

# plot two figures side by side, one is the original data points colored by cluster, the other is the new data points colored by cluster.
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(X_train[:, 0], X_train[:, 1], c=psc.predict(X_train), cmap='viridis')
axs[0].set_title('Original Data Points')
axs[1].scatter(X_new[:, 0], X_new[:, 1], c=new_labels, cmap='viridis', marker='x')
axs[1].set_title('New Data Points')
plt.show()
plt.close()
