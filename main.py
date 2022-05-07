import argparse
import sys

from train import *


def main():
    sys.path.append("../")
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_length", default=1)
    parser.add_argument("--save", default=True)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--epochs", default=3)
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--study_name", default="training")
    args = parser.parse_args()
    if args.save:
        print(f"Reading Data")
        data = read_data("./data")
        print(f"Image count {data[0].shape[0]}")
        print(f"Preprocessing")
        train_data, validation, weights = preprocessing(*data, bool(args.save), int(args.history_length))
    else:
        print(f"Reading Data")
        train_data = torch.load("./data/training")
        validation = torch.load("./data/validation")
        weights = torch.load("./data/sample_weight")
    print(f"Training")
    train_model(train_data, validation, weights, int(args.batch_size), int(args.epochs), float(args.lr),
                args.study_name, history_length=int(args.history_length))


if __name__ == "__main__":
    main()
