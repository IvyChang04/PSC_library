import pandas as pd
import torch.nn as nn
import numpy as np
import time
import datetime
import argparse
import warnings
import sklearn
from sklearn.cluster import SpectralClustering, KMeans
from ParametricSpectralClustering import PSC, Accuracy
from scipy.io import arff
from pathlib import Path
import random
import torch

r = 72
rng = np.random.RandomState(r)
torch.manual_seed(0)
random.seed(int(r))
np.random.seed(0)

ROOT = Path("JSS_Experiments").parent.absolute()

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-datasize", "--size", type=int, help="data size used for training")
parser.add_argument(
    "-methods",
    "--methods",
    nargs="+",
    help="select clustering method (psc, sc, kmeans)",
)
parser.add_argument(
    "-dataset",
    "--dataset",
    type=str,
    help="choose Letter dataset or Pendigits dataset in this experiment",
)

args = parser.parse_args()


class Net_Letter(nn.Module):
    def __init__(self) -> None:
        super(Net_Letter, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(16, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 26),
        )

    def forward(self, x):
        return self.model(x)


class Net_Firewall(nn.Module):
    def __init__(self) -> None:
        super(Net_Firewall, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(11, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.model(x)


dataset = args.dataset

if "Firewall" in dataset:
    firewall = pd.read_csv(ROOT / "datasets" / "firewall.csv")
    action = {"allow": 1, "deny": 2, "drop": 3, "reset-both": 4}
    firewall["Action"] = firewall["Action"].map(action)
    y_tmp = firewall["Action"].values
    x_tmp = firewall.drop(["Action"], axis=1).values

elif "Letter" in dataset:
    letter_arff = arff.loadarff(ROOT / "datasets" / "dataset_6_letter.arff")
    letter = pd.DataFrame(letter_arff[0])
    letter["class"] = letter["class"].astype(str)

    Class = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "J": 10,
        "K": 11,
        "L": 12,
        "M": 13,
        "N": 14,
        "O": 15,
        "P": 16,
        "Q": 17,
        "R": 18,
        "S": 19,
        "T": 20,
        "U": 21,
        "V": 22,
        "W": 23,
        "X": 24,
        "Y": 25,
        "Z": 26,
    }

    letter["class"] = letter["class"].map(Class)
    y_tmp = letter["class"].values
    x_data_tmp = letter.drop(["class"], axis=1).values

    scaler = sklearn.preprocessing.StandardScaler().fit(x_data_tmp)
    x_tmp = scaler.transform(x_data_tmp)

f = open(ROOT / "table_3" / "log.txt", "a+")
now = str(datetime.datetime.now())
f.write("======" + now + "======\n")
f.write("dataset: " + str(args.dataset) + "\n")

if args.size == -1:
    f.write("input data size: all\n")
    x_data = x_tmp
    y = y_tmp

else:
    f.write("input data size: " + str(args.size) + "\n")
    x_data = x_tmp[: args.size]
    y = y_tmp[: args.size]

scaler = sklearn.preprocessing.StandardScaler().fit(x_data)
x = scaler.transform(x_data)
methods = args.methods

kmeans_total_acc = []
kmeans_total_ari = []
kmeans_total_ami = []

sc_total_acc = []
sc_total_ari = []
sc_total_ami = []

psc_total_acc = []
psc_total_ari = []
psc_total_ami = []

result = pd.read_csv(ROOT / "table_3" / "result.csv", index_col=[0, 1])

for _ in range(10):
    if "sc" in methods:
        if "Firewall" in dataset:
            spectral_clustering = SpectralClustering(
                n_clusters=4,
                eigen_solver="arpack",
                affinity="nearest_neighbors",
                assign_labels="kmeans",
                random_state=rng,
            )
        elif "Letter" in dataset:
            spectral_clustering = SpectralClustering(
                n_clusters=26,
                eigen_solver="arpack",
                affinity="nearest_neighbors",
                assign_labels="kmeans",
                random_state=rng,
            )
        # measure time spent
        start_time = round(time.time() * 1000)
        sc_index = spectral_clustering.fit_predict(x)
        end_time = round(time.time() * 1000)

        # calculate accuracy, ari, ami
        acc = Accuracy(y_true=y, y_pred=sc_index)
        sc_accRate, sc_ari, sc_ami = acc.acc_report()

        # write result into ratio_log.txt
        f.write("---SpectralClustering---\n")
        f.write("acc rate: " + str(sc_accRate) + "\n")
        f.write("ari: " + str(sc_ari) + "\n")
        f.write("ami: " + str(sc_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n")
        sc_total_acc.append(sc_accRate)
        sc_total_ami.append(sc_ami)
        sc_total_ari.append(sc_ari)

    if "psc" in methods:
        kmeans = KMeans(n_clusters=4, init="random", n_init="auto", algorithm="elkan")
        if "Firewall" in dataset:
            psc = PSC(
                model=Net_Firewall(),
                clustering_method=kmeans,
                sampling_ratio=0,
                n_components=4,
                n_neighbor=4,
                batch_size_data=args.size,
                random_state=rng,
            )
        elif "Letter" in dataset:
            psc = PSC(
                model=Net_Letter(),
                clustering_method=kmeans,
                sampling_ratio=0,
                n_components=26,
                n_neighbor=4,
                batch_size_data=args.size,
                random_state=rng,
            )

        # measure total time spent
        start_time = round(time.time() * 1000)
        psc.fit(x)
        psc_index = psc.predict(x)
        end_time = round(time.time() * 1000)

        # calculate acc[1:m]
        acc = Accuracy(y_true=y, y_pred=psc_index)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        f.write("----------PSC-----------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n")

        psc_total_acc.append(psc_accRate)
        psc_total_ari.append(psc_ari)
        psc_total_ami.append(psc_ami)

    if "kmeans" in methods:
        if "Firewall" in dataset:
            kmeans = KMeans(
                n_clusters=4, init="random", n_init="auto", algorithm="elkan", random_state=rng,
            )
        elif "Letter" in dataset:
            kmeans = KMeans(
                n_clusters=26, init="random", n_init="auto", algorithm="elkan", random_state=rng,
            )
        start_time = round(time.time() * 1000)
        kmeans_index = kmeans.fit_predict(x)
        end_time = round(time.time() * 1000)

        acc = Accuracy(y_true=y, y_pred=kmeans_index)
        kmeans_accRate, kmeans_ari, kmeans_ami = acc.acc_report()

        f.write("----------KMeans-----------\n")
        f.write("acc rate: " + str(kmeans_accRate) + "\n")
        f.write("ari: " + str(kmeans_ari) + "\n")
        f.write("ami: " + str(kmeans_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n")

        kmeans_total_acc.append(kmeans_accRate)
        kmeans_total_ari.append(kmeans_ari)
        kmeans_total_ami.append(kmeans_ami)

if "sc" in methods:
    ari_mean, ari_std = np.mean(sc_total_ari), np.std(sc_total_ari)
    ami_mean, ami_std = np.mean(sc_total_ami), np.std(sc_total_ami)
    acc_mean, acc_std = np.mean(sc_total_acc), np.std(sc_total_acc)
    # write result into log.txt
    f.write("==============report==============\n")
    f.write("|data size: " + str(args.size) + "|\n")
    f.write("|method: Spectral Clustering|\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + "|\n")
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + "|\n")
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + "|\n")
    f.write("=====================================\n\n")
    # print result
    print("=========report=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("===========================\n\n\n")
    # write result into result.csv
    result.at[("SC", "ClusterAcc"), args.dataset] = str(acc_mean) + "±" + str(acc_std)
    result.at[("SC", "ARI"), args.dataset] = str(ari_mean) + "±" + str(ari_std)
    result.at[("SC", "AMI"), args.dataset] = str(ami_mean) + "±" + str(ami_std)

if "psc" in methods:
    ari_mean, ari_std = np.mean(psc_total_ari), np.std(psc_total_ari)
    ami_mean, ami_std = np.mean(psc_total_ami), np.std(psc_total_ami)
    acc_mean, acc_std = np.mean(psc_total_acc), np.std(psc_total_acc)
    # write result into log.txt
    f.write("==============report==============\n")
    f.write("|data size: " + str(args.size) + "|\n")
    f.write("|method: PSC|\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + "|\n")
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + "|\n")
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + "|\n")
    f.write("=====================================\n\n")
    # print result
    print("=========report=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("===========================\n\n\n")
    # write result into result.csv
    result.at[("PSC", "ClusterAcc"), args.dataset] = str(acc_mean) + "±" + str(acc_std)
    result.at[("PSC", "ARI"), args.dataset] = str(ari_mean) + "±" + str(ari_std)
    result.at[("PSC", "AMI"), args.dataset] = str(ami_mean) + "±" + str(ami_std)

if "kmeans" in methods:
    ari_mean, ari_std = np.mean(kmeans_total_ari), np.std(kmeans_total_ari)
    ami_mean, ami_std = np.mean(kmeans_total_ami), np.std(kmeans_total_ami)
    acc_mean, acc_std = np.mean(kmeans_total_acc), np.std(kmeans_total_acc)
    # write result into log.txt
    f.write("==============report==============\n")
    f.write("|data size: " + str(args.size) + "|\n")
    f.write("|method: KMeans|\n")
    f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + "|\n")
    f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + "|\n")
    f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + "|\n")
    f.write("=====================================\n\n")
    # print result
    print("=========report=========")
    print("acc:", acc_mean, "±", acc_std)
    print("ari:", ari_mean, "±", ari_std)
    print("ami:", ami_mean, "±", ami_std)
    print("===========================\n\n\n")
    # write result into result.csv
    result.at[("KMeans", "ClusterAcc"), args.dataset] = (
        str(acc_mean) + "±" + str(acc_std)
    )
    result.at[("KMeans", "ARI"), args.dataset] = str(ari_mean) + "±" + str(ari_std)
    result.at[("KMeans", "AMI"), args.dataset] = str(ami_mean) + "±" + str(ami_std)

f.close()
result.to_csv(ROOT / "table_3" / "result.csv")
