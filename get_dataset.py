import os
import csv
import time
import librosa
import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

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
        "data": np.float32(np.ones((1, num_tokens, 2 * dim4token))),
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

            filter_data = butter_bandpass_filter(data, 25.0, 400.0, 12800, order=5)

            #print(len_data)
            #print(int((len_data - frame_len) / stride + 1))

            if selection_type == "window":
                for i in range(int((len_data - frame_len) / stride + 1)):

                    t1 = data[int(i * stride * tokens4sec * dim4token) : int(i * stride * tokens4sec * dim4token + dim_frame)].reshape(1, num_tokens, -1)
                    t2 = filter_data[int(i * stride * tokens4sec * dim4token) : int(i * stride * tokens4sec * dim4token + dim_frame)].reshape(1, num_tokens, -1)

                    t = np.concatenate((t1, t2), axis=2)

                    ds["data"] = np.concatenate((ds["data"], t), axis=0)
                    ds["label"] = np.concatenate((ds["label"], file_label), axis=0)


            print(f"{file_name} done.")

    np.save("./data.npy", ds["data"][1:])
    np.save("./label.npy", ds["label"][1:])
    print(ds["data"].shape)
    print(ds["label"].shape)


if __name__ == "__main__":
    start = time.time()

    get_dataset("../datasets/pediatricPCGdataset/ten_fold/all.csv", "../datasets/pediatricPCGdataset/data", "wav", 100, 128, 1, 0.5)

    print(f"Time: {time.time() - start}s.")
    print("Done!")
