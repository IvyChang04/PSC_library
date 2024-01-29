import numpy as np
import sys
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from ParametricSpectralClustering import PSC

parser = argparse.ArgumentParser()
parser.add_argument("-data", "--train_data", type=str, help="Training data")
parser.add_argument(
    "-n_cluster", "--n_cluster", type=int, default=10, help="Number of clusters"
)
parser.add_argument("-path", "--model_path", type=str, help="The pretrained model path")
parser.add_argument(
    "-cluster_result_format",
    "--cluster_result_format",
    type=str,
    default="csv",
    help="Save the result in csv format or txt format",
)
args = parser.parse_args()


def __check_args():
    if args.train_data is not None and args.train_data[-3:] not in [
        "npy",
        "csv",
        "txt",
    ]:
        raise ValueError("The training data must be in .npy, .csv or .txt format.")
    if args.n_cluster is not None and args.n_cluster <= 0:
        raise ValueError("n_cluster must be integer and greater than 0.")
    if args.model_path is None:
        raise ValueError("Model path not assigned.")
    if args.cluster_result_format is not None and args.cluster_result_format not in [
        "csv",
        "txt",
    ]:
        raise ValueError("The saving format must be .csv or .txt.")


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

    cluster_method = KMeans(
        n_clusters=args.n_cluster,
        init="k-means++",
        n_init=1,
        max_iter=100,
        algorithm="elkan",
    )
    psc = PSC(clustering_method=cluster_method, test_splitting_rate=0)
    psc.load_model(args.model_path)
    cluster_idx = psc.predict(x)

    if args.cluster_result_format == "csv":
        df = pd.DataFrame(cluster_idx)
        df.to_csv(
            args.train_data[:-4] + "_cluster_result.csv", index=False, header=False
        )
    elif args.cluster_result_format == "txt":
        f = open(args.train_data[:-4] + "_cluster_result.txt", "w")
        np.set_printoptions(threshold=sys.maxsize)
        # print(cluster_idx)
        f.write(str(cluster_idx) + " ")
        f.close()

    print("Finished")


if __name__ == "__main__":
    main(sys.argv)
