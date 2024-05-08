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
parser.add_argument("-model_path", "--path", default=None, type=str, help="model path")
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


df = pd.read_csv(ROOT / "datasets" / "firewall.csv")
action = {"allow": 1, "deny": 2, "drop": 3, "reset-both": 4}
df["Action"] = df["Action"].map(action)
y_tmp = df["Action"].values
x_tmp = df.drop(["Action"], axis=1).values


f = open(ROOT / "Firewall_table4" / "log.txt", "a+")
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

sc_acc = []
sc_time = []

psc_acc_1_15000 = []
psc_acc_1_30000 = []
psc_acc_15001_30000 = []
psc_acc_1_45000 = []
psc_acc_15001_45000 = []
psc_acc_1_60000 = []
psc_acc_15001_60000 = []
psc_time_1_15000 = []
psc_time_1_30000 = []
psc_time_1_45000 = []
psc_time_1_60000 = []

result = pd.read_csv(ROOT / "Firewall_table4" / "result.csv")

for _ in range(10):
    if "sc" in methods:
        spectral_clustering = SpectralClustering(
            n_clusters=4,
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

        # write result into log.txt
        f.write("---SpectralClustering---\n")
        f.write("acc rate: " + str(sc_accRate) + "\n")
        f.write("ari: " + str(sc_ari) + "\n")
        f.write("ami: " + str(sc_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n")
        sc_acc.append(sc_accRate)
        sc_time.append(end_time - start_time)

    if "psc" in methods:
        kmeans = KMeans(
            n_clusters=4,
            init="random",
            n_init="auto",
            algorithm="elkan",
            random_state=rng,
        )
        psc = PSC(
            model=Net(),
            clustering_method=kmeans,
            sampling_ratio=0,
            n_components=4,
            n_neighbor=4,
            batch_size_data=args.size,
            random_state=rng,
        )

        if args.path == None:
            # measure training time spent
            train_start_time = round(time.time() * 1000)
            psc.fit(x[:15000])
            psc_index = psc.predict(x[:15000])
            train_end_time = round(time.time() * 1000)
            train_total_time = train_end_time - train_start_time
        else:
            psc.load_model(ROOT / args.path / "model.pkl")
            psc_index = psc.predict(x[:15000])
            train_total_time = np.nan

        # calculate acc on [1:15000]
        acc = Accuracy(y_true=y[:15000], y_pred=psc_index[:15000])
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        f.write("----------PSC-----------\n")
        f.write("--------PSC[1:15000]--------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        f.write("time spent: " + str(train_total_time) + "\n")
        psc_acc_1_15000.append(psc_accRate)
        psc_time_1_15000.append(train_total_time)

        # calculate time spent and acc on [1:30000]
        start_time = round(time.time() * 1000)
        psc_index_30000 = psc.predict(x[:30000])
        end_time = round(time.time() * 1000)
        acc = Accuracy(y_true=y[:30000], y_pred=psc_index_30000)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        f.write("--------PSC[1:30000]--------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        f.write("time spent: " + str(train_total_time + end_time - start_time) + "\n")
        psc_acc_1_30000.append(psc_accRate)
        psc_time_1_30000.append(train_total_time + end_time - start_time)

        # calculate acc on [15001:30000]
        acc = Accuracy(y_true=y[15001:30000], y_pred=psc_index_30000[15001:30000])
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        f.write("--------PSC[15001:30000]--------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        psc_acc_15001_30000.append(psc_accRate)

        # calculate time spent and acc on [1:45000]
        start_time = round(time.time() * 1000)
        psc_index_45000 = psc.predict(x[:45000])
        end_time = round(time.time() * 1000)
        acc = Accuracy(y_true=y[:45000], y_pred=psc_index_45000)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        f.write("--------PSC[1:45000]--------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        f.write("time spent: " + str(train_total_time + end_time - start_time) + "\n")
        psc_acc_1_45000.append(psc_accRate)
        psc_time_1_45000.append(train_total_time + end_time - start_time)

        # calculate acc[15001:45000]
        acc = Accuracy(y_true=y[15001:45000], y_pred=psc_index_45000[15001:45000])
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        f.write("--------PSC[15001:45000]--------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        psc_acc_15001_45000.append(psc_accRate)

        # calculate time spent and acc on [1:60000]
        start_time = round(time.time() * 1000)
        psc_index_60000 = psc.predict(x)
        end_time = round(time.time() * 1000)
        acc = Accuracy(y_true=y, y_pred=psc_index_60000)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        f.write("--------PSC[1:60000]--------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        f.write("time spent: " + str(train_total_time + end_time - start_time) + "\n")
        psc_acc_1_60000.append(psc_accRate)
        psc_time_1_60000.append(train_total_time + end_time - start_time)

        # calculate acc[15001:60000]
        acc = Accuracy(y_true=y[15001:60000], y_pred=psc_index_60000[15001:60000])
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        f.write("--------PSC[15001:60000]--------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        psc_acc_15001_60000.append(psc_accRate)

    if args.path != None:
        break

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

    # write result into result.txt
    result.at[(args.size / 15000), "Time"] = (
        str(sc_time_mean) + " ± " + str(sc_time_std)
    )
    result.at[(args.size / 15000), "Accuracy"] = (
        str(sc_acc_mean) + " ± " + str(sc_acc_std)
    )

if "psc" in methods:
    psc_acc_1_15000_mean = round(np.mean(psc_acc_1_15000), 3)
    psc_acc_1_15000_std = round(np.std(psc_acc_1_15000), 3)
    psc_acc_1_30000_mean = round(np.mean(psc_acc_1_30000), 3)
    psc_acc_1_30000_std = round(np.std(psc_acc_1_30000), 3)
    psc_acc_15001_30000_mean = round(np.mean(psc_acc_15001_30000), 3)
    psc_acc_15001_30000_std = round(np.std(psc_acc_15001_30000), 3)
    psc_acc_1_45000_mean = round(np.mean(psc_acc_1_45000), 3)
    psc_acc_1_45000_std = round(np.std(psc_acc_1_45000), 3)
    psc_acc_15001_45000_mean = round(np.mean(psc_acc_15001_45000), 3)
    psc_acc_15001_45000_std = round(np.std(psc_acc_15001_45000), 3)
    psc_acc_1_60000_mean = round(np.mean(psc_acc_1_60000), 3)
    psc_acc_1_60000_std = round(np.std(psc_acc_1_60000), 3)
    psc_acc_15001_60000_mean = round(np.mean(psc_acc_15001_60000), 3)
    psc_acc_15001_60000_std = round(np.std(psc_acc_15001_60000), 3)

    psc_time_1_15000_mean = round(np.mean(psc_time_1_15000), 2)
    psc_time_1_15000_std = round(np.std(psc_time_1_15000), 3)
    psc_time_1_30000_mean = round(np.mean(psc_time_1_30000), 2)
    psc_time_1_30000_std = round(np.std(psc_time_1_30000), 3)
    psc_time_1_45000_mean = round(np.mean(psc_time_1_45000), 2)
    psc_time_1_45000_std = round(np.std(psc_time_1_45000), 3)
    psc_time_1_60000_mean = round(np.mean(psc_time_1_60000), 2)
    psc_time_1_60000_std = round(np.std(psc_time_1_60000), 3)

    if args.path != None:
        psc_time_1_15000_mean = np.nan
        psc_time_1_15000_std = np.nan
        psc_time_1_30000_mean = np.nan
        psc_time_1_30000_std = np.nan
        psc_time_1_45000_mean = np.nan
        psc_time_1_45000_std = np.nan
        psc_time_1_60000_mean = np.nan
        psc_time_1_60000_std = np.nan

    # write result into log.txt
    f.write("======= PSC mean ± std =======\n")
    f.write("data size: 15000\n")
    f.write(
        "time spent: "
        + str(psc_time_1_15000_mean)
        + "±"
        + str(psc_time_1_15000_std)
        + "\n"
    )
    f.write(
        "[1:n] acc: "
        + str(psc_acc_1_15000_mean)
        + "±"
        + str(psc_acc_1_15000_std)
        + "\n"
    )
    f.write(
        "[n+1:n+m] acc: "
        + str(psc_acc_1_15000_mean)
        + "±"
        + str(psc_acc_1_15000_std)
        + "\n"
    )
    f.write(
        "all acc: "
        + str(psc_acc_1_15000_mean)
        + "±"
        + str(psc_acc_1_15000_std)
        + "\n\n"
    )
    f.write("data size: 30000\n")
    f.write(
        "time spent: "
        + str(psc_time_1_30000_mean)
        + "±"
        + str(psc_time_1_30000_std)
        + "\n"
    )
    f.write(
        "[1:n] acc: "
        + str(psc_acc_1_15000_mean)
        + "±"
        + str(psc_acc_1_15000_std)
        + "\n"
    )
    f.write(
        "[n+1:n+m] acc: "
        + str(psc_acc_15001_30000_mean)
        + "±"
        + str(psc_acc_15001_30000_std)
        + "\n"
    )
    f.write(
        "all acc: "
        + str(psc_acc_1_30000_mean)
        + "±"
        + str(psc_acc_1_30000_std)
        + "\n\n"
    )
    f.write("data size: 45000\n")
    f.write(
        "time spent: "
        + str(psc_time_1_45000_mean)
        + "±"
        + str(psc_time_1_45000_std)
        + "\n"
    )
    f.write(
        "[1:n] acc: "
        + str(psc_acc_1_15000_mean)
        + "±"
        + str(psc_acc_1_15000_std)
        + "\n"
    )
    f.write(
        "[n+1:n+m] acc: "
        + str(psc_acc_15001_45000_mean)
        + "±"
        + str(psc_acc_15001_45000_std)
        + "\n"
    )
    f.write(
        "all acc: "
        + str(psc_acc_1_45000_mean)
        + "±"
        + str(psc_acc_1_45000_std)
        + "\n\n"
    )
    f.write("data size: 60000\n")
    f.write(
        "time spent: "
        + str(psc_time_1_60000_mean)
        + "±"
        + str(psc_time_1_60000_std)
        + "\n"
    )
    f.write(
        "[1:n] acc: "
        + str(psc_acc_1_15000_mean)
        + "±"
        + str(psc_acc_1_15000_std)
        + "\n"
    )
    f.write(
        "[n+1:n+m] acc: "
        + str(psc_acc_15001_60000_mean)
        + "±"
        + str(psc_acc_15001_60000_std)
        + "\n"
    )
    f.write(
        "all acc: "
        + str(psc_acc_1_60000_mean)
        + "±"
        + str(psc_acc_1_60000_std)
        + "\n\n"
    )

    # write result into result.csv
    # data size = 15000
    result.at[1, "Time.1"] = (
        str(psc_time_1_15000_mean) + "±" + str(psc_time_1_15000_std)
    )
    result.at[1, "Accuracy.1"] = (
        str(psc_acc_1_15000_mean) + "±" + str(psc_acc_1_15000_std)
    )
    result.at[1, "Accuracy.2"] = (
        str(psc_acc_1_15000_mean) + "±" + str(psc_acc_1_15000_std)
    )
    result.at[1, "Accuracy.3"] = (
        str(psc_acc_1_15000_mean) + "±" + str(psc_acc_1_15000_std)
    )
    result.at[2, "Time.1"] = (
        str(psc_time_1_30000_mean) + "±" + str(psc_time_1_30000_std)
    )
    result.at[2, "Accuracy.1"] = (
        str(psc_acc_1_30000_mean) + "±" + str(psc_acc_1_30000_std)
    )
    result.at[2, "Accuracy.2"] = (
        str(psc_acc_1_15000_mean) + "±" + str(psc_acc_1_15000_std)
    )
    result.at[2, "Accuracy.3"] = (
        str(psc_acc_15001_30000_mean) + "±" + str(psc_acc_15001_30000_std)
    )
    result.at[3, "Time.1"] = (
        str(psc_time_1_45000_mean) + "±" + str(psc_time_1_45000_std)
    )
    result.at[3, "Accuracy.1"] = (
        str(psc_acc_1_45000_mean) + "±" + str(psc_acc_1_45000_std)
    )
    result.at[3, "Accuracy.2"] = (
        str(psc_acc_1_15000_mean) + "±" + str(psc_acc_1_15000_std)
    )
    result.at[3, "Accuracy.3"] = (
        str(psc_acc_15001_45000_mean) + "±" + str(psc_acc_15001_45000_std)
    )
    result.at[4, "Time.1"] = (
        str(psc_time_1_60000_mean) + "±" + str(psc_time_1_60000_std)
    )
    result.at[4, "Accuracy.1"] = (
        str(psc_acc_1_60000_mean) + "±" + str(psc_acc_1_60000_std)
    )
    result.at[4, "Accuracy.2"] = (
        str(psc_acc_1_15000_mean) + "±" + str(psc_acc_1_15000_std)
    )
    result.at[4, "Accuracy.3"] = (
        str(psc_acc_15001_60000_mean) + "±" + str(psc_acc_15001_60000_std)
    )


f.close()
result.to_csv(ROOT / "Firewall_table4" / "result.csv", index=False)
