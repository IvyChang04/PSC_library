import numpy as np
import sys
import argparse
import pandas as pd
from ParametricSpectralClustering import PSC


parser = argparse.ArgumentParser()
parser.add_argument("-data", "--train_data", type=str, help="Training data")
parser.add_argument(
    "-rate",
    "--test_splitting_rate",
    type=float,
    default=0.3,
    help="The splitting rate of the training data",
)
parser.add_argument(
    "-path", "--model_path", type=str, help="A string that contains the file name"
)

args = parser.parse_args()


def __check_args():
    if args.train_data is not None and args.train_data[-3:] not in [
        "npy",
        "csv",
        "txt",
    ]:
        raise ValueError("The training data must be in .npy, .csv or .txt format.")
    if args.test_splitting_rate is not None and (
        args.test_splitting_rate <= 0 or args.test_splitting_rate > 1
    ):
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

    psc = PSC(test_splitting_rate=args.test_splitting_rate)

    psc.training_psc_model(x)

    psc.save_model(args.model_path)

    print("Finished training.")


if __name__ == "__main__":
    main(sys.argv)
