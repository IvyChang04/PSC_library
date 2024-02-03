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

f = open("JSS_Experiments/Firewall_table4/log.txt", "a+")
now = str(datetime.datetime.now())
f.write("======" + now + "======")

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

sc_acc = []
sc_time = []

psc_acc_1_n = []
psc_acc_n1_m = []
psc_acc_1_m = []
psc_time_1_m = []

result = pd.read_csv("JSS_Experiments/Firewall_table4/result.csv")

for _ in range(10):
    # ---------- Spectral Clustering ----------
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
        sc_acc.append(sc_accRate)
        sc_time.append(end_time - start_time)

    # ---------- PSC ----------
    if "psc" in methods:
        kmeans = KMeans(n_clusters=4, init="random", n_init="auto", algorithm="elkan")
        # n is fixed to 15000
        sampling_ratio = 1 - (15000 / args.size)
        psc = PSC(
            model=Net(),
            clustering_method=kmeans,
            sampling_ratio=sampling_ratio,
            n_components=4,
            n_neighbor=4,
            batch_size_data=args.size,
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
        f.write("--------PSC[1:m]--------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n")
        psc_acc_1_m.append(psc_accRate)
        psc_time_1_m.append(end_time - start_time)

        # calculate acc[1:n]
        acc = Accuracy(y_true=y[:15000], y_pred=psc_index[:15000])
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        f.write("--------PSC[1:n]--------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        psc_acc_1_n.append(psc_accRate)

        # calculate acc[n+1:m]
        if args.size > 15000:
            acc = Accuracy(
                y_true=y[15001 : args.size], y_pred=psc_index[15001 : args.size]
            )
            psc_accRate, psc_ari, psc_ami = acc.acc_report()
            f.write("-------PSC[n+1:m]-------\n")
            f.write("acc rate: " + str(psc_accRate) + "\n")
            f.write("ari: " + str(psc_ari) + "\n")
            f.write("ami: " + str(psc_ami) + "\n")
            psc_acc_n1_m.append(psc_accRate)

        elif args.size == 15000:
            f.write("--------PSC[n+1:m]--------\n")
            f.write("acc rate: " + str(psc_accRate) + "\n")
            f.write("ari: " + str(psc_ari) + "\n")
            f.write("ami: " + str(psc_ami) + "\n")
            psc_acc_n1_m.append(psc_accRate)


if "sc" in methods:
    sc_acc_mean = round(np.mean(sc_acc), 3)
    sc_acc_std = round(np.std(sc_acc), 3)
    sc_time_mean = round(np.mean(sc_time), 1)
    sc_time_std = round(np.std(sc_time), 2)

    # write result into log.txt
    f.write("======= Spectral Clustering mean ± std =======\n")
    f.write("data size: " + str(args.size) + "\n")
    f.write("acc rate: " + str(sc_acc_mean) + " ± " + str(sc_acc_std) + "\n")
    f.write("time spent: " + str(sc_time_mean) + " ± " + str(sc_time_std) + "\n\n")

    # write result into result.csv
    result.at[(args.size / 15000), "Time"] = (
        str(sc_time_mean) + " ± " + str(sc_time_std)
    )
    result.at[(args.size / 15000), "Accuracy"] = (
        str(sc_acc_mean) + " ± " + str(sc_acc_std)
    )

if "psc" in methods:
    psc_acc_1_m_mean = round(np.mean(psc_acc_1_m), 3)
    psc_acc_1_m_std = round(np.std(psc_acc_1_m), 3)
    psc_acc_1_n_mean = round(np.mean(psc_acc_1_n), 3)
    psc_acc_1_n_std = round(np.std(psc_acc_1_n), 3)
    psc_acc_n1_m_mean = round(np.mean(psc_acc_n1_m), 3)
    psc_acc_n1_m_std = round(np.std(psc_acc_n1_m), 3)
    psc_time_1_m_mean = round(np.mean(psc_time_1_m), 2)
    psc_time_1_m_std = round(np.std(psc_time_1_m), 3)

    # write result into log.txt
    f.write("======= PSC mean ± std =======\n")
    f.write("data size: " + str(args.size) + "\n")
    f.write(
        "time spent: " + str(psc_time_1_m_mean) + "±" + str(psc_time_1_m_std) + "\n"
    )
    f.write("[1:n] acc: " + str(psc_acc_1_n_mean) + "±" + str(psc_acc_1_n_std) + "\n")
    f.write(
        "[n+1:m] acc: " + str(psc_acc_n1_m_mean) + "±" + str(psc_acc_n1_m_std) + "\n"
    )
    f.write("[1:m] acc: " + str(psc_acc_1_m_mean) + "±" + str(psc_acc_1_m_std) + "\n\n")

    # write result into result.csv
    result.at[(args.size / 15000), "Time.1"] = (
        str(psc_time_1_m_mean) + "±" + str(psc_time_1_m_std)
    )
    result.at[(args.size / 15000), "Accuracy.1"] = (
        str(psc_acc_1_n_mean) + "±" + str(psc_acc_1_n_std)
    )
    result.at[(args.size / 15000), "Accuracy.2"] = (
        str(psc_acc_n1_m_mean) + "±" + str(psc_acc_n1_m_std)
    )
    result.at[(args.size / 15000), "Accuracy.3"] = (
        str(psc_acc_1_m_mean) + "±" + str(psc_acc_1_m_std)
    )


f.close()
result.to_csv("JSS_Experiments/Firewall_table4/result.csv", index=False)
