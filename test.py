from ParametricSpectralClustering import Four_layer_FNN, PSC, Accuracy
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import time


def main():
    digits = load_digits()
    X = digits.data/16
    y = digits.target
    clust_method = KMeans(n_clusters=10, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
    model = Four_layer_FNN(64, 128, 256, 64, 10)
    psc = PSC(model=model, clustering_method=clust_method, test_splitting_rate=0.3)
    
    time1 = round(time.time()*1000)
    cluster_id = psc.fit_predict(X)
    time2 = round(time.time()*1000)
    print(f"time spent: {time2 - time1} milliseconds")
    acc = Accuracy(y, cluster_id)
    acc.acc_report()


if __name__ == "__main__":
    main()
