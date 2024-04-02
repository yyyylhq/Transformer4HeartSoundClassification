import os
import csv
import time
import torch
import librosa
import numpy as np
import jax.numpy as jnp
from torch.utils import data
from scipy.signal import butter, lfilter
from torchlibrosa import Spectrogram, LogmelFilterBank

def butter_bandpass(lowcut=25.0, highcut=400.0, fs=2000, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut=25.0, highcut=400.0, fs=2000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

class HeartSoundDataset(data.Dataset):

    def __init__(self, data_dir, label_file, sample_rate=32000, butter_sample_rate=2000, frame_len=1, tokens=100, stride=1):
        super(HeartSoundDataset, self).__init__()

        self.data_dir = data_dir
        self.label_file = label_file
        self.sample_rate = sample_rate
        self.butter_sample_rate = butter_sample_rate
        self.frame_len = frame_len
        self.tokens = tokens
        self.stride = stride

        #self.spec = Spectrogram(n_fft=1024, hop_length=320)

        #self.logmel = LogmelFilterBank(sr=2000, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0)

        self.spec = Spectrogram(n_fft=1024, hop_length=320, win_length=1024, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        #self.logmel = LogmelFilterBank(sr=self.sample_rate, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0, ref=1.0, amin=1e-10, freeze_parameters=True)
        self.logmel = LogmelFilterBank(sr=self.sample_rate, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)

        self.data = list()

        with open(label_file, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")

            for row in csv_reader:
                file_name = row[0] + ".wav"
                file_path = os.path.join(self.data_dir, file_name)

                data1 = torch.Tensor(librosa.load(file_path, sr=self.sample_rate)[0])

                data_t2, _ = librosa.load(file_path, sr=self.butter_sample_rate)
                data2 = torch.Tensor(butter_bandpass_filter(data_t2))

                data_t3 = data1.reshape(1, -1)

                data3 = self.spec(data_t3)

                data4 = self.logmel(data3).reshape(-1)

                #file_label = torch.Tensor([0.0, 1.0]) if int(row[1]) == 1 else torch.Tensor([1.0, 0.0])
                file_label = torch.Tensor([1]) if int(row[1]) == 1 else torch.Tensor([0])

                for i in range(int(data1.shape[0] / self.sample_rate)):
                    t1 = data1[i * self.sample_rate : i * self.sample_rate + self.sample_rate].reshape(self.tokens, -1)
                    t2 = data2[i * self.butter_sample_rate : i * self.butter_sample_rate + self.butter_sample_rate].reshape(self.tokens, -1)
                    t3 = data4[i * 6400 : i * 6400 + 6400].reshape(self.tokens, -1)

                    self.data.append([torch.cat((t1, t2, t3), dim=1), file_label])


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class HeartSoundDatasetVote(data.Dataset):

    def __init__(self, data_dir, label_file, sample_rate=32000, butter_sample_rate=2000, tokens=100, frame_len=1, stride=1):
        super(HeartSoundDatasetVote, self).__init__()

        self.data_dir = data_dir
        self.label_file = label_file
        self.sample_rate = sample_rate
        self.butter_sample_rate = butter_sample_rate
        self.frame_len = frame_len
        self.tokens = tokens
        self.stride = stride

        self.spec = Spectrogram(n_fft=1024, hop_length=320, win_length=1024, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        #self.logmel = LogmelFilterBank(sr=self.sample_rate, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0, ref=1.0, amin=1e-10, freeze_parameters=True)
        self.logmel = LogmelFilterBank(sr=self.sample_rate, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)

        self.data = list()

        with open(label_file, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")

            for row in csv_reader:
                file_name = row[0] + ".wav"
                file_path = os.path.join(self.data_dir, file_name)

                data1 = torch.Tensor(librosa.load(file_path, sr=self.sample_rate)[0])

                data_t2, _ = librosa.load(file_path, sr=self.butter_sample_rate)
                data2 = torch.Tensor(butter_bandpass_filter(data_t2))

                data_t3 = data1.reshape(1, -1)
                data3 = self.spec(data_t3)

                data4 = self.logmel(data3).reshape(-1)

                file_label = torch.Tensor([0.0, 1.0]) if int(row[1]) == 1 else torch.Tensor([1.0, 0.0])

                current_data = dict()
                current_data["file_name"] = file_name
                current_data["label"] = file_label

                l = int(data1.shape[0] / self.sample_rate)
                t1 = data1[: l * self.sample_rate].reshape(l, self.tokens, -1)
                t2 = data2[: l * self.butter_sample_rate].reshape(l, self.tokens, -1)
                t3 = data4[: l * 6400].reshape(l, self.tokens, -1)

                current_data["data"] = torch.cat((t1, t2, t3), dim=2)

                self.data.append(current_data)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    start = time.time()

    test = HeartSoundDataset("../datasets/PCCD/data/wav", "../datasets/PCCD/ten_folds/test/k0/test.csv")

    torch_test = torch.concat([test[i][0].unsqueeze(0) for i in range(len(test))], dim=0)
    ltorch_test = torch.concat([test[i][1].unsqueeze(0) for i in range(len(test))], dim=0)
    jnp_test = jnp.array(torch_test.numpy())
    ljnp_test = jnp.array(ltorch_test.numpy(), dtype=jnp.int32)

    print(f"Time: {time.time() - start}s")

