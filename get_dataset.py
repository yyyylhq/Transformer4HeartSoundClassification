import os
import csv
import time
import numpy as np
from dataset import ArrayDataset
import jax.numpy as jnp

def get_dataset(
    numpy_data_dir: str,
    csv_file: str,
    dataset_type: str = "frame"
):
    dataset_data = list()
    dataset_label = list()
    with open(csv_file, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=",")

        for row in csv_reader:
            file_name = row[0] + ".npy"
            file_path = os.path.join(numpy_data_dir, file_name)
            #print(file_name)

            file_label = np.array(1, dtype=np.int32) if int(row[1]) == 1 else np.array(0, dtype=np.int32)
            current_data = np.load(file_path)

            """
            if dataset_type == "frame":
                l = current_data.shape[0]
                labels = np.repeat(file_label, l)

                if dataset_data is None:
                    dataset_data = current_data
                else:
                    dataset_data = np.concatenate((dataset_data, current_data), axis=0)

                if dataset_label is None:
                    dataset_label = labels
                else:
                    dataset_label = np.concatenate((dataset_label, labels), axis=0)

            elif dataset_type == "vote":
                pass
            else:
                raise ValueError(f"The dataset_type must be frame or vote.")
            """
            if dataset_type == "frame":
                l = current_data.shape[0]
                labels = np.repeat(file_label, l)

                dataset_data.append(current_data)

                dataset_label.append(labels)

            elif dataset_type == "vote":
                pass
            else:
                raise ValueError(f"The dataset_type must be frame or vote.")

    #print(len(dataset_data))
    #print(len(dataset_label))
    data = [np.array(d) for d in dataset_data]
    label = [np.array(l) for l in dataset_label]
    concatenated_data = np.concatenate(data, axis=0)
    concatenated_label = np.concatenate(label, axis=0)
    #concatenated_data =  jnp.array(concatenated_data)
    #concatenated_label =  jnp.array(concatenated_label)
    #print(concatenated_data.shape)
    #print(concatenated_label.shape)

    return ArrayDataset(x=concatenated_data, y=concatenated_label)
    #return {"data": concatenated_data, "label": concatenated_label}

if __name__ == "__main__":
    start = time.time()

    #wav2numpy("../datasets/PCCD/data/wav", "../datasets/PCCD/data/npy", 100, 64, selection_type="window", frame_len=1.0, stride=1.0)
    get_dataset("../datasets/PCCD/data/npy", "../datasets/PCCD/ten_folds/train/k0/train.csv")

    print(f"Time: {time.time() - start}s.")
    print("Done!")
