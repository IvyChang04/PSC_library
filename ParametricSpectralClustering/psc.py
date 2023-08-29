import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import random
import pickle
import os

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


class Accuracy:
    """Calculate the accuracy of clustering.

    Parameters
    ----------
    y_true : list
        True labels.
    y_pred : list
        Predicted labels.
    
    Attributes
    ----------
    y_true : list
        True labels.
    y_pred : list
        Predicted labels.
    
    Examples
    --------
    >>> from PSC_lib import Accuracy
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.cluster import KMeans
    >>> digits = load_digits()
    >>> X = digits.data/16
    >>> y = digits.target
    >>> y_pred = KMeans(n_clusters=10, random_state=0).fit_predict(X)
    >>> acc = Accuracy(y, y_pred)
    >>> acc.acc_report()
    Clustering Accuracy: 0.7935447968836951
    Adjusted rand index: 0.670943009820327
    Adjusted mutual information: 0.7481788599584174
    """
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def cluster_acc(self):
        """Calculate the clustering accuracy.

        Parameters
        ----------
        self : object
            The instance itself.
        """
        self.y_true = self.y_true.astype(np.int64)
        assert self.y_pred.size == self.y_true.size
        D = max(self.y_pred.max(), self.y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(self.y_pred.size):
            w[self.y_pred[i], self.y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return w[row_ind, col_ind].sum() * 1.0 / self.y_pred.size

    def ARI(self):
        """Calculate the adjusted rand index.
        
        Parameters
        ----------
        self : object
            The instance itself.
        """
        return adjusted_rand_score(self.y_true, self.y_pred)

    def AMI(self):
        """Calculate the adjusted mutual information.

        Parameters
        ----------
        self : object
            The instance itself.
        """
        return adjusted_mutual_info_score(self.y_true, self.y_pred)

    def acc_report(self):
        """Report the accuracy of clustering.

        Parameters
        ----------
        self : object
            The instance itself.
        """
        clusterAcc = self.cluster_acc()
        ari = self.ARI()
        ami = self.AMI()

        print(f"Clustering Accuracy: {clusterAcc}")
        print(f"Adjusted rand index: {ari}")
        print(f"Adjusted mutual information: {ami}")


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
    model : torch.nn.Module
        The model used to learn the embedding.
    criterion : torch.nn.modules.loss, default=nn.MSELoss()
        The loss function used to train the model.
    epochs : int, default=50
        Number of epochs to train the model.
    clustering_method : sklearn.cluster, default=None
        The clustering method used to cluster the embedding.
    spliting_rate : float, default=0.3
        The spliting rate of the training data.
    
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
    test_splitting_rate : float
        The spliting rate of the training data.
    optimizer : torch.optim
        The optimizer used to train the model.
    epochs : int
        Number of epochs to train the model.
    clustering : str
        The clustering method used to cluster the embedding.
    model_fitted : bool
        Whether the model has been fitted.
    dataloader : torch.utils.data.DataLoader
        The dataloader used to train the model.

    Examples
    --------
    >>> from PSC_lib import PSC, Net
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.cluster import KMeans
    >>> digits = load_digits()
    >>> X = digits.data/16
    >>> cluster_method = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
    >>> model = Net(64, 128, 256, 64, 10)
    >>> psc = PSC(model=model, clustering_method=cluster_method)
    >>> psc.fit(X)
    Start training
    >>> psc.save_model("model")
    >>> cluster_idx = psc.predict(X)

    >>> from PSC_lib import PSC, Net
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.cluster import KMeans
    >>> digits = load_digits()
    >>> X = digits.data/16
    >>> cluster_method = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
    >>> model = Net(64, 128, 256, 64, 10)
    >>> psc = PSC(model=model, clustering_method=cluster_method)
    >>> psc.load_model("model")
    >>> cluster_idx = psc.predict(X)
    """
    def __init__(
        self, 
        n_neighbor = 8, 
        sigma = 1, 
        k = 10, 
        model = Net(64, 128, 256, 64, 10),
        criterion = nn.MSELoss(),
        epochs = 50,
        clustering_method = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan'),
        test_splitting_rate = 0.3
        ) -> None:

        self.n_neighbor = n_neighbor
        self.sigma = sigma
        self.k = k
        self.model = model
        self.criterion = criterion
        self.test_splitting_rate = test_splitting_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

        self.epochs = epochs
        self.clustering = clustering_method
        self.model_fitted = False

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

    def __loss_calculation(self):
        running_loss = 0.0

        for inputs, labels in self.dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(self.dataloader)

    def __train_model(self, X, x):
        self.model_fitted = True
        print("Start training")
        u = torch.from_numpy(self.__spectral_clustering(X)).type(torch.FloatTensor)
        dataset = torch.utils.data.TensorDataset(x, u)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 50, shuffle = True)
        self.dataloader = dataloader

        for _ in range(self.epochs):
            loss = self.__loss_calculation()
            if(loss < 0.00015):
                break

    def __check_file_exist(self, file_name) -> bool:
        for entry in os.listdir('./'):
            if entry == file_name:
                return True
        return False

    def __check_clustering_method(self) -> None:
        if self.clustering is None:
            raise ValueError(
                "No clustering method assigned."
            )

    def __check_model(self) -> None:
        if self.model is None:
            raise ValueError(
                "No model assigned."
            )

    def training_psc_model(self, X):
        self.__check_clustering_method()
        self.__check_model()

        x = torch.from_numpy(X).type(torch.FloatTensor)

        if self.test_splitting_rate == 0:
            X_train, x_train = X, x
        
        else:
            X_train, _, x_train, _ = train_test_split(
                X, x, test_size=self.test_splitting_rate, random_state=random.randint(1, 100))

        self.__train_model(X_train, x_train)

        U = self.model(x).detach().numpy()

        return U

    def fit(self, X):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.        
        """
        U = self.training_psc_model(X)
        
        if hasattr(self.clustering, "fit") is False:
            raise AttributeError(
                f"'{type(self.clustering)}' object has no attribute 'fit'"
            )

        self.clustering.fit(U)

        return self

    def fit_predict(self, X):
        """Fit the model according to the given training data and predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        Returns
        -------
        cluster_index : array-like of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        U = self.training_psc_model(X)

        if hasattr(self.clustering, "fit_predict") is False:
            raise AttributeError(
                f"'{type(self.clustering)}' object has no attribute 'fit_predict'"
            )

        return self.clustering.fit_predict(U)

    # predict the closest cluster
    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.
        
        Returns
        -------
        cluster_index : array-like of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """

        x = torch.from_numpy(X).type(torch.FloatTensor)
        U = self.model(x).detach().numpy()
        if hasattr(self.clustering, "predict") is False:
            raise AttributeError(
                f"'{type(self.clustering)}' object has no attribute 'predict'"
            )

        if self.model_fitted is False:
            return self.clustering.fit_predict(U)

        return self.clustering.predict(U)
        
    def set_model(self, self_defined_model) -> None:
        """Set the model to a self-defined model.
        
        Parameters
        ----------
        self_defined_model : torch.nn.Module
            The self-defined model.
        """
        
        self.model = self_defined_model

    def save_model(self, path: str) -> None:
        """Save the model to a file.

        Parameters
        ----------
        path : str
            The path of the file.
        """
        torch.save(self.model.state_dict(), path)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path: str) -> None:
        """Load the model from a file.

        Parameters
        ----------
        path : str
            The path of the file.
        """
        if self.__check_file_exist(path) is False:
            raise FileNotFoundError(
                f"No such file or directory: '{path}'"
            )
        
        with open(path, 'rb') as f:
                self.model = pickle.load(f)
