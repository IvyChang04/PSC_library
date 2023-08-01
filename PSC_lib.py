import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import pickle
import os


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output):
        super().__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.predict = nn.Linear(n_hidden3, n_output)
        
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)
        return x


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


class PSC:

    def __init__(
        self, 
        n_neighbor = 8, 
        sigma = 1, 
        k = 10, 
        model = Net(64, 128, 256, 64, 10),
        criterion = nn.MSELoss(),
        epochs = 50,
        clustering = "kmeans",
        name = None
        ) -> None:

        self.n_neighbor = n_neighbor
        self.sigma = sigma
        self.k = k
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        self.epochs = epochs
        self.clustering = clustering
        self.name = name

    # input 轉換成做kmeans之前的matrix
    def __spectral_clustering(self, X):
        dist  = cdist(X, X, "euclidean")
        S = np.zeros(dist.shape)
        neighbor_index = np.argsort(dist, axis = 1)[:, 1:self.n_neighbor + 1]

        for i in range(X.shape[0]):
            S[i, neighbor_index[i]] = np.exp(-dist[i, neighbor_index[i]] / (2 * self.sigma ** 2))

        S = np.maximum(S, S.T)
        D = np.diag(np.sum(S, axis = 1))
        L = D - S
        D_tmp = np.linalg.inv(D) ** (1 / 2)
        L_sym = np.dot(np.dot(D_tmp, L), D_tmp)
        A,B=np.linalg.eig(L_sym)
        idx=np.argsort(A)[:self.k]
        U=B[:,idx]
        U=U/((np.sum(U**2,axis=1)**0.5)[:,None])
        return U


    def __train(self):
    
        running_loss = 0.0

        for inputs, labels in self.dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(self.dataloader)

    def __check_file_exist(self, file_name) -> bool:
        for entry in os.listdir('./'):
            if entry == file_name:
                return True
        return False

    def fit(self, X, saving_path = None, use_existing_model = None):
        x = torch.from_numpy(X).type(torch.FloatTensor)

        if saving_path is not None and use_existing_model is not None:
            raise ValueError(
                "Can't save model and load model at the same time!"
            )
            
        elif use_existing_model is not None:
            # check if model path exists
            if self.__check_file_exist(use_existing_model) is False:
                raise ValueError(
                    f"File `{use_existing_model}` do not exist"
                )

            print("Using existing model")
            with open(self.name, 'rb') as f:
                self.model = pickle.load(f)

        elif use_existing_model is None:

            print("Start training")
            u = torch.from_numpy(self.__spectral_clustering(X)).type(torch.FloatTensor)
            dataset = torch.utils.data.TensorDataset(x, u)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = 50, shuffle = True)
            self.dataloader = dataloader

            for _ in range(self.epochs):
                loss = self.__train()
                if(loss < 0.00015):
                    break

                if saving_path is not None:
                    torch.save(self.model.state_dict(), saving_path)
                    with open(self.name, 'wb') as f:
                        pickle.dump(self.model, f)
                    # self.model_exist = True

        U = self.model(x).detach().numpy()

        if isinstance(self.clustering, str) and self.clustering == "kmeans":
            kmeans = KMeans(n_clusters=self.k, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
            cluster = kmeans.fit(U)

        elif isinstance(self.clustering, str) and self.clustering == "dbscan":
            dbscan = DBSCAN()
            cluster = dbscan.fit(U)

        return cluster

        """
        digits = load_digits()
        X = digits.data/16

        cluster_index = PSC(clustering = "kmeans").fit(X)
        """

    def fit_predict(self, X, saving_path = None):
        return self.fit(X, saving_path).labels_

    # predict the closest cluster
    def predict(self, X, sample_weight = 'deprecate'):
        x = torch.from_numpy(X).type(torch.FloatTensor)
        u = torch.from_numpy(self.__spectral_clustering(X)).type(torch.FloatTensor)
        dataset = torch.utils.data.TensorDataset(x, u)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 50, shuffle = True)
        self.dataloader = dataloader

        for _ in range(self.epochs):
            loss = self.__train()
            if(loss < 0.00015):
                break

        U = self.model(x).detach().numpy()

        if isinstance(self.clustering, str) and self.clustering == "kmeans":
            kmeans = KMeans(n_clusters=self.k, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
            cluster = kmeans.predict(U)


    def set_model(self, self_defined_model):
        self.model = self_defined_model

def main():
    # data
    digits = load_digits()
    X = digits.data/16

    # saving_path = "Spectraal_Clustering"
    cluster_index = PSC().fit_predict(X)

    print(cluster_index)

if __name__ == "__main__":
    main()
