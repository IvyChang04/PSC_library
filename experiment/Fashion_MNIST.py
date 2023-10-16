import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import time
import random
from psc import PSC, Accuracy, Four_layer_FNN

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 7, 1, 3),
        )

    def forward(self, x, stop=False):
        x = self.encoder(x)
        if stop:
            x = x.view(x.size(0), -1)
            return x
        x = self.decoder(x)
        return x

class Dimen_Reduct_Model(nn.Module):
    def __init__(self):
        super(Dimen_Reduct_Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(49, 196),
            nn.ReLU(inplace=False),
            nn.Linear(196, 392),
            nn.ReLU(inplace=False),
            nn.Linear(392, 196),
            nn.ReLU(inplace=False),
            nn.Linear(196, 10)
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x


train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
train_loader = DataLoader(train_data, batch_size=24, shuffle=True)
test_loader = DataLoader(test_data, batch_size=24, shuffle=True)

autoencoder = Autoencoder()
autoencoder = torch.load("experiment\autoencoder.pth")

train_num = 60000
cut_1 = 30000

X_train = train_data.data[:train_num]/255
X_train = X_train.unsqueeze(1) 
X_train = X_train.to(torch.float32)

X_train_1 = autoencoder(X_train[:cut_1], stop=True).detach().numpy()
X_train_2 = autoencoder(X_train[cut_1:], stop=True).detach().numpy()
X_train = np.concatenate((X_train_1, X_train_2), axis=0)

y_train = train_data.targets[:train_num].numpy()
print("finish data transformation")

kmeans = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
sc = SpectralClustering(n_clusters=10, assign_labels='discretize', random_state=0)
model = Four_layer_FNN(49, 196, 392, 196, 10)
psc = PSC(model=model, clustering_method=sc, test_splitting_rate=0.7)

# kmeans
print("\nkmeans")
time1 = round(time.time()*1000)
Kmeans_index = kmeans.fit_predict(X_train)
time2 = round(time.time()*1000)
print(f"Kmeans time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train, y_pred=Kmeans_index)
acc.acc_report()

# spectral clustering: can't run with 60000 data
# print("\nspectral clustering")
# time1 = round(time.time()*1000)
# SC_index = sc.fit_predict(X_train)
# time2 = round(time.time()*1000)
# print(f"SC time spent: {time2 - time1} milliseconds")

# acc = Accuracy(y_true=y_train, y_pred=SC_index)
# acc.acc_report()

# PSC + sc
# print("\nPSC + sc")
# time1 = round(time.time()*1000)
# PSC_index_1 = psc.fit_predict(X_train_1)
# time2 = round(time.time()*1000)
# print(f"PSC time spent: {time2 - time1} milliseconds")
# acc = Accuracy(y_true=y_train[:cut_1], y_pred=PSC_index_1)
# acc.acc_report()

# time1 = round(time.time()*1000)
# PSC_index_2 = psc.predict(X_train_2)
# time2 = round(time.time()*1000)
# print(f"PSC time spent: {time2 - time1} milliseconds")

# acc = Accuracy(y_true=y_train[cut_1:], y_pred=PSC_index_2)
# acc.acc_report()

psc = PSC(model=model, clustering_method=kmeans, test_splitting_rate=0.7)

# PSC + kmeans
print("\nPSC + kmeans")
print("\nfit_predict() for 30000 data")
time1 = round(time.time()*1000)
PSC_index_1 = psc.fit_predict(X_train_1)
time2 = round(time.time()*1000)
print(f"PSC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[:cut_1], y_pred=PSC_index_1)
acc.acc_report()

print("\npredict() for 30000 data")
time1 = round(time.time()*1000)
PSC_index_2 = psc.predict(X_train_2)
time2 = round(time.time()*1000)
print(f"PSC time spent: {time2 - time1} milliseconds")

acc = Accuracy(y_true=y_train[cut_1:], y_pred=PSC_index_2)
acc.acc_report()

print("\naccuracy rate for 60000 data")
result = np.concatenate((PSC_index_1, PSC_index_2), axis=0)
acc = Accuracy(y_true=y_train, y_pred=result)
acc.acc_report()