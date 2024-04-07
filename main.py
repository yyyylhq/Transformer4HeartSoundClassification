import logging
import argparse
from train import train

if __name__ == "__main__":

    tf = "k0"

    batch_size = 256
    epochs = 500
    lr = 0.0001

    seed = 2479

    tokens = 100
    input_dim = 404


    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")

    parser_train.add_argument("--train_dataset", type=str, default="./label/PCCD/0/train.csv")
    parser_train.add_argument("--val_dataset", type=str, default="./label/PCCD/0/val.csv")
    parser_train.add_argument("--test_dataset", type=str, default="./label/PCCD/0/val.csv")
    parser_train.add_argument("--tf", type=str, default="k0")

    parser_train.add_argument("--epochs", type=int, default=200)
    parser_train.add_argument("--batch_size", type=int, default=256)
    parser_train.add_argument("--learning_rate", type=float, default=0.0001)
    parser_train.add_argument("--momentum", type=float, default=0.9)

    parser_train.add_argument("--params_rng", type=int, default=0)
    parser_train.add_argument("--dropout_rng", type=int, default=1)

    args = parser.parse_args()
    print(args)

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        pass
    else:
        raise Exception('Error argument!')
