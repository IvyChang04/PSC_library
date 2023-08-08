import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import pickle
import os
from sklearn.utils.fixes import threadpool_limits

"""
TODO:
- add other clustering methods
    AgglomerativeClustering
    GaussianMixture
- function comments(documentation) 
    Done
- add `fit` and `predict` examples in PSC() class
    Done
"""


class Net(nn.Module):
    """The model used to learn the embedding.
    
    Parameters
    ----------
    n_feature : int
        The number of input features.
    n_hidden1 : int
        The number of neurons in the first hidden layer.
    n_hidden2 : int
        The number of neurons in the second hidden layer.
    n_hidden3 : int
        The number of neurons in the third hidden layer.
    n_output : int
        The number of output features.
    
    Attributes
    ----------
    hidden1 : torch.nn.Linear
        The first hidden layer.
    hidden2 : torch.nn.Linear
        The second hidden layer.
    hidden3 : torch.nn.Linear
        The third hidden layer.
    predict : torch.nn.Linear
        The output layer.
    
    Examples
    --------
    >>> model = Net(64, 128, 256, 64, 10)
    >>> model
    Net(
        (hidden1): Linear(in_features=64, out_features=128, bias=True)
        (hidden2): Linear(in_features=128, out_features=256, bias=True)
        (hidden3): Linear(in_features=256, out_features=64, bias=True)
        (predict): Linear(in_features=64, out_features=10, bias=True)
    )
    """
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output):
        super().__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.predict = nn.Linear(n_hidden3, n_output)
        
    def forward(self, x):
        """Forward propagation.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)
        return x

def cluster_acc(y_true, y_pred):
    """Calculate clustering accuracy. Require scikit-learn installed.

    Parameters
    ----------
    y_true : list
        True labels.
    y_pred : list
        Predicted labels.
    
    Returns
    -------
    accuracy : float
        clustering accuracy
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


class PSC:
    """Parametric Spectral Clustering.

    Parameters
    ----------
    n_neighbor : int, default=8
        Number of neighbors to use when constructing the adjacency matrix using k-nearest neighbors.
    sigma : float, default=1
        The sigma value for the Gaussian kernel.
    k : int, default=10
        Number of clusters.
    model : torch.nn.Module, default=Net(64, 128, 256, 64, 10)
        The model used to learn the embedding.
    criterion : torch.nn.modules.loss, default=nn.MSELoss()
        The loss function used to train the model.
    epochs : int, default=50
        Number of epochs to train the model.
    clustering : str, default="kmeans"
        The clustering method used to cluster the embedding.
    name : str, default=None
        The name of the model file to save.
    
    Attributes
    ----------
    n_neighbor : int
        Number of neighbors to use when constructing the adjacency matrix using k-nearest neighbors.
    sigma : float
        The sigma value for the Gaussian kernel.
    k : int
        Number of clusters.
    model : torch.nn.Module
        The model used to learn the embedding.
    criterion : torch.nn.modules.loss
        The loss function used to train the model.
    optimizer : torch.optim
        The optimizer used to train the model.
    epochs : int
        Number of epochs to train the model.
    clustering : str
        The clustering method used to cluster the embedding.
    name : str
        The name of the model file to save.
    dataloader : torch.utils.data.DataLoader
        The dataloader used to train the model.
    cluster : sklearn.cluster
        The clustering model used to cluster the embedding.
    
    Examples
    --------
    >>> from PSC_lib import PSC, Net
    >>> from sklearn.datasets import load_digits
    >>> digits = load_digits()
    >>> X = digits.data/16
    >>> model = Net(64, 128, 256, 64, 10)
    >>> cluster_index = PSC(model = model).fit_predict(X)
    Start training
    >>> cluster_index
    array([5, 2, 2, ..., 2, 8, 2], dtype=int32)
    >>> clust = PSC(model = model).fit(X, saving_path = "Spectral_Clustering")
    Start training
    >>> clust = PSC(model = model).fit(X, use_existing_model = "Spectral_Clustering")
    Using existing model
    >>> clust.predict(X, model = "Spectral_Clustering")
    array([3, 9, 9, ..., 9, 8, 9])

    
    >>> model = Net(64, 128, 256, 64, 10)
    >>> PSC(model = model).fit_predict(X)
    array([0, 1, 2, ..., 8, 9, 8], dtype=int32)
    >>> PSC(model = model).fit(X)
    KMeans(algorithm='elkan', copy_x=True, init='k-means++', max_iter=100,
        n_clusters=10, n_init=1, n_jobs=None, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)
    >>> PSC(model = model).predict(X)
    array([0, 1, 2, ..., 8, 9, 8], dtype=int32)
    >>> PSC(model = model).set_model(model)
    """
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
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        saving_path : str, default=None
            The name of the model file to save.
        use_existing_model : str, default=None
            The name of the model file to load.
        
        Returns
        -------
        self : object
            Returns self.        
        """
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
            self.name = use_existing_model
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
                    self.name = saving_path
                    with open(self.name, 'wb') as f:
                        pickle.dump(self.model, f)
                    # self.model_exist = True

        U = self.model(x).detach().numpy()

        if isinstance(self.clustering, str) and self.clustering == "kmeans":
            kmeans = KMeans(n_clusters=self.k, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
            cluster = kmeans.fit(U)
        elif isinstance(self.clustering, str) and self.clustering == "gaussian_mixture":
            gmm = GaussianMixture(n_components=self.k)
            cluster = gmm.fit(U)
        elif isinstance(self.clustering, str) and self.clustering == "agglomerative":
            agglomerative = AgglomerativeClustering(n_clusters=self.k, linkage='ward')
            cluster = agglomerative.fit(U)
        self.cluster = cluster
        
        return self

    def fit_predict(self, X, saving_path = None):
        """Fit the model according to the given training data and predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        saving_path : str, default=None
            The name of the model file to save.
        
        Returns
        -------
        cluster_index : array-like of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if isinstance(self.clustering, str) and self.clustering == "kmeans":
            return self.fit(X, saving_path).cluster.labels_
        elif isinstance(self.clustering, str) and self.clustering == "gaussian_mixture":
            return self.fit(X, saving_path).cluster.means_
        elif isinstance(self.clustering, str) and self.clustering == "agglomerative":
            return self.fit(X, saving_path).cluster.labels_

    # predict the closest cluster
    def predict(self, X, model = None):
        """Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        model : str, default=None
            The name of the model file to load.
        
        Returns
        -------
        cluster_index : array-like of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.__check_file_exist(model) is False:
            ValueError(
                f"model {model} does not exist"
            )
        elif model is None:
            ValueError(
                "model cannot be None"
            )

        self.name = model
        with open(self.name, 'rb') as f:
                self.model = pickle.load(f)

        x = torch.from_numpy(X).type(torch.FloatTensor)
        U = self.model(x).detach().numpy()

        if isinstance(self.clustering, str) and self.clustering == "kmeans":
            return self.cluster.predict(U)
        elif isinstance(self.clustering, str) and self.clustering == "gaussian_mixture":
            return self.cluster.predict(U)
        elif isinstance(self.clustering, str) and self.clustering == "agglomerative":
            return self.cluster.fit_predict(U)

    def set_model(self, self_defined_model):
        """Set the model to a self-defined model.
        
        Parameters
        ----------
        self_defined_model : torch.nn.Module
            The self-defined model.
        """
        self.model = self_defined_model

def main():
    # data
    digits = load_digits()
    X = digits.data/16

    # saving_path = "Spectral_Clustering"
    # cluster_index = PSC().fit_predict(X)

    # kmeans
    print("kmeans")
    clust = PSC().fit(X, use_existing_model='kmeans')
    print(type(clust))
    clust.predict(X, model = 'kmeans')
    print(clust.cluster.cluster_centers_) # Coordinates of cluster centers.
    
    # gaussian_mixture
    print("gaussian")
    clust = PSC(clustering='gaussian_mixture').fit(X, use_existing_model='gaussian_mixture')
    print(type(clust))
    clust.predict(X, model = 'gaussian_mixture')
    print(clust.cluster.means_) # The mean of each mixture component.

    # agglomerative
    print("agglomerative")
    clust = PSC(clustering='agglomerative').fit(X, use_existing_model='agglomerative')
    print(type(clust))
    clust.predict(X, model = 'agglomerative')
    print(clust.cluster.labels_) # Cluster labels for each point.

if __name__ == "__main__":
    main()
