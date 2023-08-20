import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


train_num = 1000
train_dataset = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=False)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
        )


    def forward(self, x, stop=False):
        x = self.encoder(x)
        if stop:
            return x
        x = self.decoder(x)
        return x

model = Autoencoder()
device = torch.device("mps")
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    running_loss = 0.0
    for x,y in train_loader:
        x = x.to(device)
        x = x.view(-1, 28 * 28)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        Loss = running_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.4f}")


with torch.no_grad():
    test_images, _ = next(iter(test_loader))
    x_test = test_images.view(-1, 28 * 28)
    reconstructed = model(x_test)

    plt.figure(figsize=(20, 4))

    for i in range(10):
        # Original Images
        plt.subplot(2, 10, i + 1)
        plt.imshow(test_images[i][0], cmap='gray')
        plt.axis('off')

        # Reconstructed Images
        plt.subplot(2, 10, i + 11)
        plt.imshow(reconstructed[i].view(28, 28), cmap='gray')
        plt.axis('off')

    plt.show()


X_train = train_dataset.data[:train_num]/255
X_train = X_train.view(-1, 28 * 28)
X_train = model(X_train, stop=True).detach().numpy()
y_train = train_dataset.targets[:train_num].numpy()


k = 10
def SC(X):
    #Similarity Matrix
    dist = cdist(X,X,"euclidean")
    n_neigh=10
    S=np.zeros(dist.shape)
    neigh_index=np.argsort(dist,axis=1)[:,1:n_neigh+1]
    sigma=1
    for i in range(X.shape[0]):
        S[i,neigh_index[i]]=np.exp(-dist[i,neigh_index[i]]/(2*sigma**2))
    S=np.maximum(S,S.T)
    k=10

    #Normalized spectral clustering according to Ng, Jordan, and Weiss
    D=np.diag(np.sum(S,axis=1))
    L=D-S
    D_tmp=np.linalg.inv(D)**(1/2)
    L_sym=np.dot(np.dot(D_tmp,L),D_tmp)
    A,B=np.linalg.eig(L_sym)
    idx=np.argsort(A)[:k]
    U=B[:,idx]
    U=U/((np.sum(U**2,axis=1)**0.5)[:,None])
    return U


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


U = SC(X_train)

#K-means
kmeans = KMeans(n_clusters=k, init='random', n_init='auto', max_iter=300, algorithm='elkan')
cluster1_index = kmeans.fit_predict(U)
acc1 = cluster_acc(y_train, cluster1_index)
print(f"SC acc: {acc1:.3f}")