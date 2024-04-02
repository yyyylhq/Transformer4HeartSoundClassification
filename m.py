import torch
import time
import logging
from torch.utils.data import DataLoader
from configs.models.transformerEncoderConfig import TransformerEncoderConfig
from models.wav2vec import Wav2Vec
import torch.nn.functional as F
from hsdataset import HeartSoundDataset, HeartSoundDatasetVote

#@hydra.main(version_base=None, config_path="./conf", config_name="config")
#def main(cfg : DictConfig):
#    t = TransformerEncoderConfig(cfg["TransformerEncoderConfig"])
#    print(t)

if __name__ == "__main__":
    batch_size = 256
    epochs = 500
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.0001
    tf = "k3"
    tokens = 100
    input_dim = 404
    layers = 6
    seed = 2479


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

    torch.manual_seed(seed)

    dataset_loaded_time_start = time.time()

    #train_dataset = HeartSoundDataset("../datasets/pediatricPCGdataset/data/wav", f"../datasets/pediatricPCGdataset/ten_fold/{tf}/train/train.csv")
    #test_dataset = HeartSoundDataset("../datasets/pediatricPCGdataset/data/wav", f"../datasets/pediatricPCGdataset/ten_fold/{tf}/test/test.csv")
    #test_vote_dataset = HeartSoundDatasetVote("../datasets/pediatricPCGdataset/data/wav", f"../datasets/pediatricPCGdataset/ten_fold/{tf}/test/test.csv")


    #train_dataset = HeartSoundDataset("../datasets/PCCD/data/wav", f"../datasets/PCCD/ten_folds/train/{tf}/train.csv")
    test_dataset = HeartSoundDataset("../datasets/PCCD/data/wav", f"../datasets/PCCD/ten_folds/test/{tf}/test.csv")
    test_vote_dataset = HeartSoundDatasetVote("../datasets/PCCD/data/wav", f"../datasets/PCCD/ten_folds/test/{tf}/test.csv")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Dataset loaded time: {(time.time() - dataset_loaded_time_start):.2f}s.")

    model = Wav2Vec(tokens=tokens, input_dim=input_dim, seed=seed, num_layers=layers, device=device).to(device)
    #model = Wav2Vec(tokens=tokens, device=device).to(device)
    #model = torch.load("weights/pretrain.pth")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    max_train_acc: float = 0.0
    max_test_acc: float = 0.0
    max_test_vote_acc: float = 0.0

    train_time_start = time.time()

    for i in range(epochs):
        epoch_time_start = time.time()

        logger.info(f"-------------------------------")
        logger.info(f"Train epoch {i + 1}/{epochs} start.")

        model.train()
        ten_batch_loss: float = 0.0
        correct = 0
        for batch, (X, y) in enumerate(train_dataloader):
            #b = X.shape[0]
            #X, y = X.reshape(b, tokens, -1).to(device), y.to(device)
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = F.binary_cross_entropy_with_logits(pred[0], y)
            pred = F.softmax(model(X)[0], dim=1)
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ten_batch_loss += loss.item()

            if (batch + 1) % 10 == 0:
                logger.info(f"Batch {batch + 1}/{len(train_dataloader)} done, loss: {ten_batch_loss:.6f}.")
                ten_batch_loss = 0.0

        logger.info(f"Train epoch {i + 1} end, time: {(time.time() - epoch_time_start):.2f}s.")
        logger.info(f"Train val accuracy: {(correct / len(train_dataset)):>0.6f}")
        if correct / len(train_dataset) > max_train_acc:
            max_train_acc = correct / len(train_dataset)

        model.eval()

        correct = 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(test_dataloader):
                #b = X.shape[0]
                #X, y = X.reshape(b, tokens, -1).to(device), y.to(device)
                X, y = X.to(device), y.to(device)
                pred = F.softmax(model(X)[0], dim=1)
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        logger.info(f"Test val accuracy: {(correct / len(test_dataset)):>0.6f}")
        if correct / len(test_dataset) > max_test_acc:
            max_test_acc = correct / len(test_dataset)

        correct = 0
        with torch.no_grad():
            for i in range(len(test_vote_dataset)):
                X = test_vote_dataset[i]["data"].to(device)
                b = X.shape[0]
                #X = X.reshape(b, tokens, -1).to(device)
                y = test_vote_dataset[i]["label"].expand(b, -1).to(device)

                pred = F.softmax(model(X)[0], dim=1)
                cor = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
                if cor / b > 0.5:
                    correct += 1

            logger.info(f"Test vote accuracy: {(correct / len(test_vote_dataset)):>0.6f}")
            if correct / len(test_vote_dataset) > max_test_vote_acc:
                max_test_vote_acc = correct / len(test_vote_dataset)

        logger.info(f"Max train acc: {max_train_acc:>0.6f}, Max test acc: {max_test_acc:>0.6f}, Max test vote acc: {max_test_vote_acc:>0.6f}.")

    logger.info(f"Train end time: {(time.time() - train_time_start):.2f}s.")


    torch.save(model, f"./weights/{tf}_500.pth")
    torch.save(optimizer, f"./weights/{tf}_opt_500.pth")
