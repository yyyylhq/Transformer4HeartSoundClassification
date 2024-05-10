import os
import time
import torch
import librosa
import numpy as np
from scipy.signal import butter, lfilter
from torchlibrosa import Spectrogram, LogmelFilterBank

def butter_bandpass_filter(
    data,
    lowcut: float,
    highcut: float,
    fs: int,
    order: int = 2
):

    def _butter_bandpass(lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a

    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)

    return y

def wav2numpy(
    wav_data_dir: str,
    numpy_save_dir: str,
    tokensofsec: int,
    dim4token: int,
    selection_type="window",
    frame_len: float = 1.0,
    stride: float = 1.0
):
    spec = Spectrogram(n_fft=127, hop_length=64, win_length=64, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
    spec_t = Spectrogram(n_fft=1024, hop_length=64, win_length=64, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)

    logmel = LogmelFilterBank(sr=25600, n_fft=1024, n_mels=64, fmin=25.0, fmax=14000.0, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)

    wav_data_dir_list = [item for item in os.listdir(wav_data_dir) if item.endswith(".wav")]


    num_tokens = int(frame_len * tokensofsec)
    dim_frame = num_tokens * dim4token


    for file in wav_data_dir_list:
        file_path = os.path.join(wav_data_dir, file)

        ds = np.empty((1, int(frame_len * tokensofsec), int(2 * dim4token)))

        orig_data, sr = librosa.load(file_path, sr=tokensofsec * dim4token)

        len_data = orig_data.shape[0] / sr

        filter_data = butter_bandpass_filter(orig_data, 25.0, 400.0, int(tokensofsec * dim4token), order=2)

        filter_data_t = torch.Tensor(filter_data.reshape(1, -1))
        spec_data = np.array(spec(filter_data_t)).reshape(-1)

        spec_data_t = spec_t(filter_data_t)
        mfcc_data = np.array(logmel(spec_data_t)).reshape(-1)

        ds = None
        if selection_type == "window":
            for i in range(int((len_data - frame_len) / stride + 1)):

                t1 = orig_data[int(i * stride * tokensofsec * dim4token) : int(i * stride * tokensofsec * dim4token + dim_frame)].reshape(1, num_tokens, -1)
                t2 = filter_data[int(i * stride * tokensofsec * dim4token) : int(i * stride * tokensofsec * dim4token + dim_frame)].reshape(1, num_tokens, -1)
                t3 = spec_data[int(i * stride * tokensofsec * dim4token) : int(i * stride * tokensofsec * dim4token + dim_frame)].reshape(1, num_tokens, -1)
                t4 = mfcc_data[int(i * stride * tokensofsec * dim4token) : int(i * stride * tokensofsec * dim4token + dim_frame)].reshape(1, num_tokens, -1)

                t = np.concatenate((t1, t2, t3, t4), axis=2)
                if ds is None:
                    ds = t
                else:
                    ds = np.concatenate((ds, t), axis=0)

        np.save(os.path.join(numpy_save_dir, file[:-4]), ds)
        print(f"{file_path} Done.")


if __name__ == "__main__":
    start = time.time()

    wav2numpy("../datasets/PCCD/data/wav", "../datasets/PCCD/data/npy", 100, 64, selection_type="window", frame_len=1.0, stride=1.0)
    #get_dataset("../datasets/PCCD/data/npy", "../datasets/PCCD/ten_folds/train/k0/train.csv")

    print(f"Time: {time.time() - start}s.")
    print("Done!")
