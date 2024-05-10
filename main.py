import jax
import argparse
#from absl import app, flags, logging
#from clu import platform, metric_writers
#from ml_collections import config_flags

import train

"""
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    #logging.info(f"JAX process: {jax.process_index()} / {jax.process_count()}")
    #logging.info(f"JAX local devices: {jax.local_devices()}")
    logging.set_verbosity(logging.INFO)
    logdir = "./log"
    writer = metric_writers.create_default_writer(logdir)
    for step in range(10):
        writer.write_scalars(step, dict(loss=0.9**step))

    #train.train_and_eval(FLAGS.config)



"""

if __name__ == "__main__":
    """
    flags.mark_flag_as_required("config")

    app.run(main)
    """

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

    parser_train.add_argument("--train_dataset", type=str, default="../datasets/PCCD/ten_folds/train/k0/train.csv")
    parser_train.add_argument("--val_dataset", type=str, default="../datasets/PCCD/ten_folds/test/k0/test.csv.csv")
    parser_train.add_argument("--test_dataset", type=str, default="../datasets/PCCD/ten_folds/test/k0/test.csv")
    parser_train.add_argument("--tf", type=str, default="k0")

    parser_train.add_argument("--epochs", type=int, default=200)
    parser_train.add_argument("--batch_size", type=int, default=256)
    parser_train.add_argument("--learning_rate", type=float, default=0.0001)

    parser_train.add_argument("--params_rng", type=int, default=0)
    parser_train.add_argument("--dropout_rng", type=int, default=1)

    args = parser.parse_args()
    print(args)

    if args.mode == "train":
        train.train(args)
    elif args.mode == "test":
        pass
    else:
        raise Exception('Error argument!')
