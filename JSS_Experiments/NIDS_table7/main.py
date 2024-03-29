import pandas as pd
import torch.nn as nn
import sklearn
from sklearn.cluster import SpectralClustering, KMeans
from ParametricSpectralClustering import PSC, Accuracy
import time
import datetime
import argparse
import warnings
import numpy as np

# python NIDS/exp.py --methods kmeans --size -1 --rate 0.9

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-datasize", "--size", type=int, help="data size used for training")
parser.add_argument("-methods", "--methods", nargs="+", help="which method to test")
parser.add_argument("-sampling_ratio", "--ratio", type=float, help="sampling ratio")
args = parser.parse_args()


class Net_emb(nn.Module):
    def __init__(self) -> None:
        super(Net_emb, self).__init__()
        self.fc1 = nn.Linear(42, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.output_layer(x)
        return x


df = pd.read_csv("JSS_Experiments/datasets/NF-UQ-NIDS-v2.csv", nrows=1040000)
Class = {
    "Benign": 1,
    "DDoS": 2,
    "DoS": 3,
    "scanning": 4,
    "Reconnaissance": 4,
    "xss": 4,
    "password": 4,
    "injection": 4,
    "Bot": 4,
    "Brute Force": 4,
    "Infilteration": 4,
    "Exploits": 4,
    "Fuzzers": 4,
    "Backdoor": 4,
    "Generic": 4,
    "mitm": 4,
    "ransomware": 4,
    "Analysis": 4,
    "Theft": 4,
    "Shellcode": 4,
    "Worms": 4,
}
df["Attack"] = df["Attack"].map(Class)
print(df.drop(["Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Dataset"], axis=1).info())

y_tmp = df["Attack"].values
x_tmp = df.drop(["Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Dataset"], axis=1).values

f = open("JSS_Experiments/NIDS_table7/log.txt", "a+")
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

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 10)).fit(x_data)
x = scaler.transform(x_data)
methods = args.methods

total_acc = []
total_time = []
total_ari = []
total_ami = []

result = pd.read_csv("JSS_Experiments/NIDS_table7/result.csv", index_col=0)

for i in range(10):
    # --------Spectral Clustering--------
    if "sc" in methods:
        spectral_clustering = SpectralClustering(
            n_clusters=10,
            eigen_solver="arpack",
            affinity="nearest_neighbors",
            assign_labels="kmeans",
        )

        # measure time spent
        start_time = round(time.time() * 1000)
        sc_index = spectral_clustering.fit_predict(x)
        end_time = round(time.time() * 1000)

        # calculate acc, ari, ami
        acc = Accuracy(y_true=y, y_pred=sc_index)
        sc_accRate, sc_ari, sc_ami = acc.acc_report()
        total_acc.append(sc_accRate)
        total_ari.append(sc_ari)
        total_ami.append(sc_ami)
        total_time.append(end_time - start_time)

        # write result into log.txt
        f.write("---SpectralClustering---\n")
        f.write("acc rate: " + str(sc_accRate) + "\n")
        f.write("ari: " + str(sc_ari) + "\n")
        f.write("ami: " + str(sc_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n")

    # --------kmeans--------
    if "kmeans" in methods:
        kmeans = KMeans(n_clusters=10, init="random", n_init="auto", algorithm="elkan")

        # measure time spent
        start_time = round(time.time() * 1000)
        kmeans_index = kmeans.fit_predict(x)
        end_time = round(time.time() * 1000)

        # calculate acc, ari, ami
        acc = Accuracy(y_true=y, y_pred=kmeans_index)
        kmeans_accRate, kmeans_ari, kmeans_ami = acc.acc_report()
        total_acc.append(kmeans_accRate)
        total_ari.append(kmeans_ari)
        total_ami.append(kmeans_ami)
        total_time.append(end_time - start_time)

        # write result into log.txt
        f.write("---------Kmeans---------\n")
        f.write("acc rate: " + str(kmeans_accRate) + "\n")
        f.write("ari: " + str(kmeans_ari) + "\n")
        f.write("ami: " + str(kmeans_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n")

    # --------Parametric Spectral Clustering--------
    if "psc" in methods:
        model = Net_emb()
        kmeans = KMeans(n_clusters=10, init="random", n_init="auto", algorithm="elkan")
        psc = PSC(
            model=model,
            clustering_method=kmeans,
            sampling_ratio=args.ratio,
            n_components=10,
            n_neighbor=10,
            batch_size_data=args.size,
        )

        # measure time spent
        start_time = round(time.time() * 1000)
        psc.fit(x)
        psc_index = psc.predict(x)
        end_time = round(time.time() * 1000)

        # save model
        psc.save_model("NIDS/psc_model" + str(i + 1) + ".pkl")

        # calculate acc, ari, ami
        acc = Accuracy(y_true=y, y_pred=psc_index)
        psc_accRate, psc_ari, psc_ami = acc.acc_report()
        total_acc.append(psc_accRate)
        total_ari.append(psc_ari)
        total_ami.append(psc_ami)
        total_time.append(end_time - start_time)

        # write result into log.txt
        f.write("----------PSC----------\n")
        f.write("acc rate: " + str(psc_accRate) + "\n")
        f.write("ari: " + str(psc_ari) + "\n")
        f.write("ami: " + str(psc_ami) + "\n")
        f.write("time spent: " + str(end_time - start_time) + "\n\n\n")

# compute time, acc, ari, ami mean±std
ari_mean, ari_std = np.mean(total_ari), np.std(total_ari)
ami_mean, ami_std = np.mean(total_ami), np.std(total_ami)
acc_mean, acc_std = np.mean(total_acc), np.std(total_acc)
time_mean, time_std = np.mean(total_time), np.std(total_time)

# write mean±std into log.txt
f.write("==============report==============\n")
f.write("|data size: " + str(args.size) + "|\n")
f.write("|method: " + str(args.methods) + "|\n")
f.write("|acc: " + str(acc_mean) + "±" + str(acc_std) + "|\n")
f.write("|ari: " + str(ari_mean) + "±" + str(ari_std) + "|\n")
f.write("|ami: " + str(ami_mean) + "±" + str(ami_std) + "|\n")
f.write("|time: " + str(time_mean) + "±" + str(time_std) + "|\n")
f.write("=====================================\n\n")

print("=========report=========")
print("acc:", acc_mean, "±", acc_std)
print("ari:", ari_mean, "±", ari_std)
print("ami:", ami_mean, "±", ami_std)
print("time:", time_mean, "±", time_std)
print("===========================\n\n\n")


# write result into result.csv
if "psc" in methods:
    result.at[str(args), "PSC"] = str(time_mean) + "±" + str(time_std)
    result.at[str(args), "PSC.1"] = str(acc_mean) + "±" + str(acc_std)
    result.at[str(args), "PSC.2"] = str(ari_mean) + "±" + str(ari_std)
    result.at[str(args), "PSC.3"] = str(ami_mean) + "±" + str(ami_std)

if "sc" in methods:
    result.at[str(args), "SC"] = str(time_mean) + "±" + str(time_std)
    result.at[str(args), "SC.1"] = str(acc_mean) + "±" + str(acc_std)
    result.at[str(args), "SC.2"] = str(ari_mean) + "±" + str(ari_std)
    result.at[str(args), "SC.3"] = str(ami_mean) + "±" + str(ami_std)

f.close()
df.to_csv("JSS_Experiments/NIDS_table7/result.csv")
