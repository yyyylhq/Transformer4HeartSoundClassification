import time
import torch
import hsdataset
import numpy as np

if __name__ == "__main__":
    start = time.time()
    for i in range(10):
        train_save_data_name = f"../datasets/PCCD/ten_folds/train/k{i}/train_data.npy"
        train_save_label_name = f"../datasets/PCCD/ten_folds/train/k{i}/train_label.npy"

        test_save_data_name = f"../datasets/PCCD/ten_folds/test/k{i}/test_data.npy"
        test_save_label_name = f"../datasets/PCCD/ten_folds/test/k{i}/test_label.npy"

        train = hsdataset.HeartSoundDataset(f"../datasets/PCCD/data/wav", f"../datasets/PCCD/ten_folds/train/k{i}/train.csv")
        test = hsdataset.HeartSoundDataset(f"../datasets/PCCD/data/wav", f"../datasets/PCCD/ten_folds/test/k{i}/test.csv")

        
        torch_train_data = torch.concat([train[i][0].unsqueeze(0) for i in range(len(train))], dim=0)
        torch_train_label = torch.concat([train[i][1].unsqueeze(0) for i in range(len(train))], dim=0)
        np_train_data = np.array(torch_train_data.numpy())
        np_train_label = np.array(torch_train_label.numpy(), dtype=np.int32)

        np.save(train_save_data_name, np_train_data)
        np.save(train_save_label_name, np_train_label)

        torch_test_data = torch.concat([test[i][0].unsqueeze(0) for i in range(len(test))], dim=0)
        torch_test_label = torch.concat([test[i][1].unsqueeze(0) for i in range(len(test))], dim=0)
        np_test_data = np.array(torch_test_data.numpy(), dtype=np.float32)
        np_test_label = np.array(torch_test_label.numpy(), dtype=np.int32)

        np.save(test_save_data_name, np_test_data)
        np.save(test_save_label_name, np_test_label)

    print(f"Time: {time.time() - start}s.")
