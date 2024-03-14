import time
import logging

if __name__ == "__main__":
    tf = "k0"

    batch_size = 256
    epochs = 500
    lr = 0.0001

    seed = 9527

    tokens = 100
    input_dim = 404


    file_handler = logging.FileHandler(f"./log/log_{tf}.txt")
    console_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.DEBUG)

    file_handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s: %(message)s"))
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(asctime)s: %(message)s"))

    logger = logging.getLogger(f"log_{tf}")
    logger.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

