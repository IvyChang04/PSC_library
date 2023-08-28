import numpy as np
import sys
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from ParametricSpectralClustering import PSC


parser = argparse.ArgumentParser()
parser.add_argument("-data", "--train_data", type=str,
                    help="Training data")
parser.add_argument("-n_cluster", "--n_cluster", type=int,
                    help="Number of clusters")
parser.add_argument("-rate", "--test_splitting_rate", type=float,
                    help="The splitting rate of the training data")
args = parser.parse_args()


def __check_args():
    if args.train_data is not None and args.train_data[-3:] not in ["npy", "csv", "txt"]:
        raise ValueError(
            "The training data must be in .npy, .csv or .txt format."
        )
    if args.n_cluster is not None and args.n_cluster <= 0:
        raise ValueError(
            "n_cluster must be integer and greater than 0."
        )
    if args.test_splitting_rate is not None and (args.test_splitting_rate <= 0 or args.test_splitting_rate > 1):
        raise ValueError(
            "Test_splitting_rate must be floating point and between 0 and 1."
        )    

def __load_data():
    if args.train_data[-3:] == "npy":
        # load from npy
        x = np.load(args.train_data)
    elif args.train_data[-3:] == "csv":
        # load from csv
        df = pd.read_csv(args.train_data, header=None)
        x = df.to_numpy()
    elif args.train_data[-3:] == "txt":
        # load from txt
        x = np.loadtxt(args.train_data, dtype=int)

    return x

def main(argv):
    __check_args()

    x = __load_data()

    if args.n_cluster is not None:
        cluster_method = KMeans(n_clusters=args.n_cluster, init="k-means++", n_init=1, max_iter=100, algorithm='elkan')
        psc = PSC(clustering_method=cluster_method, test_splitting_rate=args.test_splitting_rate)
        cluster_idx = psc.fit_predict(x)
    else:
        psc = PSC(test_splitting_rate=args.test_splitting_rate)
        cluster_idx = psc.fit_predict(x)

    # save to csv
    df = pd.DataFrame(cluster_idx)
    df.to_csv(args.train_data[:-4]+"_cluster_result.csv", index=False, header=False)

    # save to txt
    f = open(args.train_data[:-4]+"_cluster_result.txt", "w")
    np.set_printoptions(threshold=sys.maxsize)
    print(cluster_idx)
    f.write(str(cluster_idx) + " ")
    f.close()

if __name__ == "__main__":
    main(sys.argv)