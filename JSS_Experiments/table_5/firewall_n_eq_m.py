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

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-datasize", "--size", type=int, help="data size used for training")
parser.add_argument(
    "-methods",
    "--methods",
    nargs="+",
    help="select clustering method (psc, sc, kmeans)",
)
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(11, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.model(x)


df = pd.read_csv("JSS_Experiments/datasets/firewall.csv")
action = {"allow": 1, "deny": 2, "drop": 3, "reset-both": 4}
df["Action"] = df["Action"].map(action)
y_tmp = df["Action"].values
x_tmp = df.drop(["Action"], axis=1).values

f = open("JSS_Experiments/table5/log.txt", "a+")
now = str(datetime.datetime.now())
f.write("======" + now + "======\n")

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

sc_total_acc = []
sc_total_ari = []
sc_total_ami = []

psc_total_acc = []
psc_total_ari = []
psc_total_ami = []

result = pd.read_csv("JSS_Experiments/table_5/result.csv", index_col=0)

for _ in range(10):
    if "sc" in methods:
        spectral_clustering = SpectralClustering(
            n_clusters=4,
            eigen_solver="arpack",
            affinity="nearest_neighbors",
            assign_labels="kmeans",
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
        sc_total_ari.append(sc_ari)
        sc_total_ami.append(sc_ami)

    if "psc" in methods:
        kmeans = KMeans(n_clusters=4, init="random", n_init="auto", algorithm="elkan")

        psc = PSC(
            model=Net(),
            clustering_method=kmeans,
            sampling_ratio=0,
            n_components=4,
            n_neighbor=4,
            batch_size_data=args.size,
        )

        # measure total time spent
        start_time = round(time.time() * 1000)
        psc.fit(x)
        psc_index = psc.predict(x)
        end_time = round(time.time() * 1000)

        # calculate acc, ari, ami
        acc = Accuracy(y_true=y, y_pred=psc_index)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()

        # write result into log.txt
        f.write("----------PSC-----------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n")

        psc_total_acc.append(psc_accRate)
        psc_total_ari.append(psc_ari)
        psc_total_ami.append(psc_ami)


# calculate mean±std
if "sc" in methods:
    ari_mean, ari_std = np.mean(sc_total_ari), np.std(sc_total_ari)
    ami_mean, ami_std = np.mean(sc_total_ami), np.std(sc_total_ami)
    acc_mean, acc_std = np.mean(sc_total_acc), np.std(sc_total_acc)
    # write result into log.txt
    f.write("==============report==============\n")
    f.write("|data size: " + str(args.size) + "|\n")
    f.write("|method: " + str(args.methods) + "|\n")
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
    result.at[str(args), "SC"] = str(acc_mean) + "±" + str(acc_std)
    result.at[str(args), "SC.1"] = str(ari_mean) + "±" + str(ari_std)
    result.at[str(args), "SC.2"] = str(ami_mean) + "±" + str(ami_std)

if "psc" in methods:
    ari_mean, ari_std = np.mean(psc_total_ari), np.std(psc_total_ari)
    ami_mean, ami_std = np.mean(psc_total_ami), np.std(psc_total_ami)
    acc_mean, acc_std = np.mean(psc_total_acc), np.std(psc_total_acc)
    # write result inot log.txt
    f.write("==============report==============\n")
    f.write("|data size: " + str(args.size) + "|\n")
    f.write("|method: " + str(args.methods) + "|\n")
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
    result.at[str(args), "PSC"] = str(acc_mean) + "±" + str(acc_std)
    result.at[str(args), "PSC.1"] = str(ari_mean) + "±" + str(ari_std)
    result.at[str(args), "PSC.2"] = str(ami_mean) + "±" + str(ami_std)

f.close()
result.to_csv("JSS_Experiments/table_5/result.csv")
