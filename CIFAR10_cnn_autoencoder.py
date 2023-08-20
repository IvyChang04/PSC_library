import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


train_num = 1000
train_dataset = datasets.CIFAR10(root='data', train=True, download=False, transform=ToTensor())
test_dataset = datasets.CIFAR10(root='data', train=False, download=False, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 1, 5, 1, 2),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 3, 5, 1, 2),
        )


    def forward(self, x, stop=False):
        x = self.encoder(x)
        if stop:
            x = x.view(x.size(0), -1)
            return x        
        x = self.decoder(x)
        return x


model = Autoencoder()
# device = torch.device("mps")
# model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):
    running_loss = 0.0
    for x,y in train_loader:
        # x = x.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        Loss = running_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.4f}")


# original and reconstructed images
num_images_to_display = 10  # Number of images to display

with torch.no_grad():
    for x, _ in test_loader:
        reconstructed = model(x)  # Get the reconstructed images

        # Convert tensors back to numpy arrays for visualization
        original_images = x.numpy()
        reconstructed_images = reconstructed.numpy()

        # Plot the images
        plt.figure(figsize=(20, 4))
        
        for i in range(num_images_to_display):
            plt.subplot(2, num_images_to_display, i + 1)
            plt.imshow(np.transpose(original_images[i], (1, 2, 0)))
            plt.axis('off')

            plt.subplot(2, num_images_to_display, num_images_to_display + i + 1)
            plt.imshow(np.transpose(reconstructed_images[i], (1, 2, 0)))
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        break


k = 10
def SC(X):
    #Similarity Metrix
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


def cluster_acc(y, y_pred):
    y = y.astype(np.int64)
    assert y_pred.size == y.size
    D = max(y_pred.max(), y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y[i]] += 1
    row_idx, col_idx = linear_sum_assignment(w.max() - w)
    return w[row_idx, col_idx].sum() * 1.0 / y_pred.size


X_train = train_dataset.data[:train_num]/ 255.0
X_train = X_train.transpose((0, 3, 1, 2))
X_train = torch.tensor(X_train, dtype=torch.float32)
X_train = model(X_train, stop=True).detach().numpy()
y_train = train_dataset.targets[:train_num]
y_train = np.array(y_train)
U = SC(X_train)


# K-means
kmeans = KMeans(n_clusters=k, init='random', n_init='auto', max_iter=300, algorithm='elkan')
cluster1_index = kmeans.fit_predict(U)
sc_acc = cluster_acc(y_train, cluster1_index)
print(f"SC acc: {sc_acc:.3f}")
