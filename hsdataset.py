import os
import csv
import time
import torch
import librosa
import numpy as np
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
        self.logmel = LogmelFilterBank(sr=self.sample_rate, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)

        self.data = list()

        with open(label_file, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")

            for row in csv_reader:
                file_name = row[0] + ".wav"
                file_path = os.path.join(self.data_dir, file_name)

                #data1 = librosa.load(file_path, sr=self.sample_rate)[0]
                data1 = librosa.load(file_path, sr=self.sample_rate)[0]
                #data1 = torch.Tensor(librosa.load(file_path, sr=self.sample_rate)[0])

                data_t2, _ = librosa.load(file_path, sr=self.butter_sample_rate)
                data2 = butter_bandpass_filter(data_t2)

                data_t3 = torch.Tensor(data1[np.newaxis, :])
                #print(data_t3.shape)
                #print(data1.shape)
                #print(data2.shape)

                data3 = self.spec(data_t3)

                data4 = self.logmel(data3).reshape(-1)

                data4 = data4.numpy()

                #file_label = torch.Tensor([1.0, 0.0]) if int(row[1]) == -1 else torch.Tensor([0.0, 1.0])
                #file_label_sure = torch.Tensor([1.0]) if int(row[2]) == 1 else torch.Tensor([0.0])

                file_label = np.zeros(1, dtype=np.int32) if int(row[1]) == -1 else np.ones(1, dtype=np.int32)
                #file_label_sure = torch.Tensor([1.0]) if int(row[2]) == 1 else torch.Tensor([0.0])

                for i in range(int(data1.shape[0] / self.sample_rate)):
                    t1 = data1[i * self.sample_rate : i * self.sample_rate + self.sample_rate].reshape(self.tokens, -1)
                    t2 = data2[i * self.butter_sample_rate : i * self.butter_sample_rate + self.butter_sample_rate].reshape(self.tokens, -1)
                    t3 = data4[i * 6400 : i * 6400 + 6400].reshape(self.tokens, -1)

                    #self.data.append([torch.cat((t1, t2, t3), dim=1), file_label, file_label_sure])
                    self.data.append([np.concatenate((t1, t2, t3), axis=1), file_label])


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
        self.logmel = LogmelFilterBank(sr=self.sample_rate, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)

        self.data = list()

        with open(label_file, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")

            for row in csv_reader:
                file_name = row[0] + ".wav"
                file_path = os.path.join(self.data_dir, file_name)

                data1 = librosa.load(file_path, sr=self.sample_rate)[0]

                data_t2, _ = librosa.load(file_path, sr=self.butter_sample_rate)
                data2 = butter_bandpass_filter(data_t2)

                data_t3 = torch.Tensor(data1.reshape(1, -1))
                data3 = self.spec(data_t3)

                data4 = self.logmel(data3).reshape(-1).numpy()

                #file_label = torch.Tensor([1.0, 0.0]) if int(row[1]) == -1 else torch.Tensor([0.0, 1.0])
                file_label = np.zeros(1, dtype=np.int32) if int(row[1]) == -1 else np.ones(1, dtype=np.int32)
                #file_label_sure = torch.Tensor([1.0]) if int(row[2]) == 1 else torch.Tensor([0.0])

                current_data = dict()

                l = int(data1.shape[0] / self.sample_rate)
                t1 = data1[: l * self.sample_rate].reshape(l, self.tokens, -1)
                t2 = data2[: l * self.butter_sample_rate].reshape(l, self.tokens, -1)
                t3 = data4[: l * 6400].reshape(l, self.tokens, -1)

                current_data["file_name"] = file_name
                current_data["data"] = np.concatenate((t1, t2, t3), axis=2)
                current_data["label"] = file_label

                self.data.append(current_data)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    start = time.time()
    #train = HeartSoundDataset("../datasets/HS-PCCC2016/data", "../datasets/HS-PCCC2016/train/train.csv")
    #test = HeartSoundDataset("../datasets/HS-PCCC2016/data", "../datasets/HS-PCCC2016/test/test.csv")
    testvote = HeartSoundDatasetVote("../datasets/HS-PCCC2016/data", "../datasets/HS-PCCC2016/test/test.csv")
    d = testvote[0]
    print(*d)
    print(d["file_name"])
    print(d["data"].shape)
    print(d["label"])
    print(time.time()- start)

    """
    all = len(train)
    positive: int = 0
    negative: int = 0
    for i in range(all):
        if int(train[i][1][0]) == 1:
            positive += 1

        if int(train[i][1][1]) == 1:
            negative += 1

    print(all, positive, negative, positive / all)

    test = HeartSoundDataset("../datasets/HS-PCCC2016/test", "../datasets/HS-PCCC2016/test/test.csv")

    all = len(test)
    positive: int = 0
    negative: int = 0
    for i in range(all):
        if int(train[i][1][0]) == 1:
            positive += 1

        if int(train[i][1][1]) == 1:
            negative += 1

    print(all, positive, negative, positive / all)

    train = HeartSoundDataset("../datasets/HS-PCCC2016/data", "../datasets/HS-PCCC2016/train/k1/train.csv")
    test = HeartSoundDataset("../datasets/HS-PCCC2016/data", "../datasets/HS-PCCC2016/test/k1/test.csv")
    print(train[0][0].shape)
    print(len(test))
    print(len(train))

    train = HeartSoundDatasetVote("../datasets/HS-PCCC2016/data", "../datasets/HS-PCCC2016/train/k1/train.csv")
    test = HeartSoundDatasetVote("../datasets/HS-PCCC2016/data", "../datasets/HS-PCCC2016/test/k1/test.csv")
    print(train[0]["data"].shape)
    print(len(test))
    print(len(train))
    """
