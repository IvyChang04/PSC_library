import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import random
import pickle
import os


class Four_layer_FNN(nn.Module):
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
    >>> model = Four_layer_FNN(64, 128, 256, 64, 10)
    >>> model
    Four_layer_FNN(
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
    >>> from ParametricSpectralClustering import Accuracy
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

        Returns
        -------
        acc : float
            The clustering accuracy.
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

        Returns
        -------
        ari : float
            The adjusted rand index.
        """
        return adjusted_rand_score(self.y_true, self.y_pred)

    def AMI(self):
        """Calculate the adjusted mutual information.

        Parameters
        ----------
        self : object
            The instance itself.

        Returns
        -------
        ami : float
            The adjusted mutual information.
        """
        return adjusted_mutual_info_score(self.y_true, self.y_pred)

    def acc_report(self):
        """Report the accuracy of clustering.

        Parameters
        ----------
        self : object
            The instance itself.

        Returns
        -------
        clusterAcc : float
            The clustering accuracy.
        ari : float
            The adjusted rand index.
        ami : float
            The adjusted mutual information.
        """
        clusterAcc = self.cluster_acc()
        ari = self.ARI()
        ami = self.AMI()
        return clusterAcc, ari, ami


class PSC:
    """Parametric Spectral Clustering.

    Parameters
    ----------
    n_neighbor : int, default=8
        Number of neighbors to use when constructing the adjacency matrix using k-nearest neighbors.
    n_clusters : int, default=10
        Number of clusters.
    model : torch.nn.Module
        The model used to learn the embedding.
    criterion : torch.nn.modules.loss, default=nn.MSELoss()
        The loss function used to train the model.
    epochs : int, default=50
        Number of epochs to train the model.
    sampling_ratio : float, default=0.3
        The spliting rate of the testing data.
    batch_size_data : int, default=50
        The batch size of the training data.
    batch_size_dataloader : int, default=20
        The batch size of the dataloader.
    clustering_method : sklearn.cluster, default=None
        The clustering method used to cluster the embedding.
    n_components : int, default=0
        The number of embedding dimensions.
    random_state : int, default=None
        The random state.

    Attributes
    ----------
    n_neighbor : int
        Number of neighbors to use when constructing the adjacency matrix using k-nearest neighbors.
    n_clusters : int
        Number of clusters.
    model : torch.nn.Module
        The model used to learn the embedding.
    criterion : torch.nn.modules.loss
        The loss function used to train the model.
    sampling_ratio : float
        The spliting rate of the testing data.
    optimizer : torch.optim
        The optimizer used to train the model.
    clustering : str
        The clustering method used to cluster the embedding.
    n_components : int
        The number of embedding dimensions.
    random_state : int
        The random state.
    epochs : int
        Number of epochs to train the model.
    model_fitted : bool
        Whether the model has been fitted.
    batch_size_data : int
        The batch size of the training data.
    batch_size_dataloader : int
        The batch size of the dataloader.
    dataloader : torch.utils.data.DataLoader
        The dataloader used to train the model.


    Examples
    --------
    >>> from ParametricSpectralClustering import PSC, Four_layer_FNN
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.cluster import KMeans
    >>> digits = load_digits()
    >>> X = digits.data/16
    >>> cluster_method = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
    >>> model = Four_layer_FNN(64, 128, 256, 64, 10)
    >>> psc = PSC(model=model, clustering_method=cluster_method, n_neighbor=10, sampling_ratio=0, batch_size_data=1797)
    >>> psc.fit(X)
    >>> psc.save_model("model")
    >>> cluster_idx = psc.predict(X)

    >>> from ParametricSpectralClustering import PSC, Four_layer_FNN
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.cluster import KMeans
    >>> digits = load_digits()
    >>> X = digits.data/16
    >>> cluster_method = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
    >>> model = Four_layer_FNN(64, 128, 256, 64, 10)
    >>> psc = PSC(model=model, clustering_method=cluster_method)
    >>> psc.load_model("model")
    >>> cluster_idx = psc.predict(X)
    """

    def __init__(
        self,
        n_neighbor=8,
        n_clusters=10,
        model=Four_layer_FNN(64, 128, 256, 64, 10),
        criterion=nn.MSELoss(),
        epochs=50,
        sampling_ratio=0.3,
        batch_size_data=50,
        batch_size_dataloader=20,
        clustering_method=None,
        n_components=0,
        random_state=None,
    ) -> None:
        self.n_neighbor = n_neighbor
        self.n_clusters = n_clusters
        self.model = model
        self.criterion = criterion
        self.sampling_ratio = sampling_ratio
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if random_state is None:
            self.random_state = random.randint(1, 100)
        else:
            self.random_state = random_state

        if clustering_method is None:
            self.clustering = KMeans(
                n_clusters=self.n_clusters,
                init="k-means++",
                n_init=1,
                max_iter=100,
                algorithm="elkan",
                random_state=self.random_state,
            )
        else:
            self.clustering = clustering_method

        if n_components == 0:
            self.n_components = self.n_clusters
        else:
            self.n_components = n_components

        self.epochs = epochs
        self.model_fitted = False

        self.batch_size_data = batch_size_data
        self.batch_size_dataloader = batch_size_dataloader

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
        connectivity = kneighbors_graph(
            X, n_neighbors=self.n_neighbor, include_self=False
        )
        affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        embedding = spectral_embedding(
            affinity_matrix_,
            n_components=self.n_components,
            eigen_solver="arpack",
            random_state=1,
            eigen_tol="auto",
            drop_first=False,
        )
        u = torch.from_numpy(embedding).type(torch.FloatTensor)
        dataset = torch.utils.data.TensorDataset(x, u)
        # not sure whether shuffle = True will affect the result
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size_dataloader, shuffle=False
        )
        self.dataloader = dataloader
        total_loss = 0
        for _ in range(self.epochs):
            loss = self.__loss_calculation()
            total_loss += loss
        return total_loss / self.epochs

    def __check_file_exist(self, file_name) -> bool:
        return os.path.exists(file_name)

    def __check_clustering_method(self) -> None:
        if self.clustering is None:
            raise ValueError("No clustering method assigned.")

    def __check_model(self) -> None:
        if self.model is None:
            raise ValueError("No model assigned.")

    def fit(self, X):
        """Train a model to convert input features into spectral embeddings.

        Parameters
        ----------
        X : array-like of shape
            Training data.

        Returns
        -------
        No return value
        """

        self.__check_clustering_method()
        self.__check_model()

        x = torch.from_numpy(X).type(torch.FloatTensor)

        if self.sampling_ratio >= 1 or self.sampling_ratio < 0:
            raise AttributeError(
                f"'test_spliting_rate' should be not less than 0 and less than 1."
            )

        if self.sampling_ratio == 0:
            X_train, x_train = X, x

        else:
            X_train, _, x_train, _ = train_test_split(
                X,
                x,
                test_size=self.sampling_ratio,
                random_state=self.random_state,
            )

        batch_size = self.batch_size_data
        total_loss = 0
        i = 1
        # Train the model in mini-batches
        for start_idx in range(0, len(X_train), batch_size):
            end_idx = start_idx + batch_size
            X_batch = X_train[start_idx:end_idx]
            x_batch = x_train[start_idx:end_idx]
            loss = self.__train_model(X_batch, x_batch)
            total_loss += loss
            if i % 20 == 0:
                total_loss = 0
            i += 1

    def fit_predict(self, X):
        """Fit the model and predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        cluster_index : array-like of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(X)

        if hasattr(self.clustering, "fit_predict") is False:
            raise AttributeError(
                f"'{type(self.clustering)}' object has no attribute 'fit_predict'"
            )

        return self.predict(X)

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

        # turn input data points into low-dim embedding
        emb = self.model(x).detach().numpy()

        if hasattr(self.clustering, "fit_predict") is False:
            raise AttributeError(
                f"'{type(self.clustering)}' object has no attribute 'predict'"
            )

        return self.clustering.fit_predict(emb)

    def save_model(self, path: str) -> None:
        """Save the model to a file.

        Parameters
        ----------
        path : str
            The path of the file.
        """

        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path: str) -> None:
        """Load the model from a file.

        Parameters
        ----------
        path : str
            The path of the file.
        """
        if self.__check_file_exist(path) is False:
            raise FileNotFoundError(f"No such file or directory: '{path}'")

        with open(path, "rb") as f:
            self.model = pickle.load(f)
