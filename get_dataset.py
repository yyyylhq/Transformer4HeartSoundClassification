import os
import csv
import time
import librosa
import numpy as np
#import jax.numpy as jnp

def get_dataset(
    csv_label_file,
    data_dir,
    file_type,
    tokens4sec: int,
    dim4token: int,
    frame_len,
    stride,
    selection_type="window"
):
    data_dir = os.path.join(data_dir, file_type)
    print(data_dir)

    num_tokens = frame_len * tokens4sec
    dim_frame = num_tokens * dim4token

    ds = {
        "data": np.float32(np.ones((1, num_tokens, dim4token))),
        "label": np.int16(np.ones(1))
    }
    print(ds["data"].shape)
    print(ds["label"].shape)

    with open(csv_label_file, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=",")

        for row in csv_reader:
            file_name = row[0] + "." + file_type
            file_label = np.int16(np.zeros(1)) if int(row[1]) == -1 else np.int16(np.ones(1))

            file_path = os.path.join(data_dir, file_name)
            data, sr = librosa.load(file_path, sr=tokens4sec * dim4token)

            len_data = data.shape[0] / sr

            #print(len_data)
            #print(int((len_data - frame_len) / stride + 1))

            if selection_type == "window":
                for i in range(int((len_data - frame_len) / stride + 1)):

                    t = data[int(i * stride * tokens4sec * dim4token) : int(i * stride * tokens4sec * dim4token + dim_frame)].reshape(1, num_tokens, -1)

                    ds["data"] = np.concatenate((ds["data"], t), axis=0)
                    ds["label"] = np.concatenate((ds["label"], file_label), axis=0)


            print(f"{file_name} done.")

    np.save("./data.npy", ds["data"][1:])
    np.save("./label.npy", ds["label"][1:])
    print(ds["data"].shape)
    print(ds["label"].shape)


if __name__ == "__main__":
    start = time.time()

    get_dataset("../datasets/HS-PCCC2016/train/k0/train.csv", "../datasets/HS-PCCC2016/data", "wav", 100, 128, 1, 0.5)

    print(f"Time: {time.time() - start}s.")
    print("Done!")
