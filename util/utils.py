import importlib
import time
import sys
import os
import json
import torch
from pesq import pesq
import numpy as np
from pystoi.stoi import stoi
from torch.utils import data
from torchmetrics import ScaleInvariantSignalDistortionRatio

import scipy.signal as sg

from segan_data_preprocess import serialized_test_folder, serialized_train_folder


def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        print('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)


def normalize(x):
    max_peak = np.max(np.abs(x))
    return x / max_peak


def snr_db(signal, noise):
    return 10.0*np.log10(np.sum(np.square(signal))/np.sum(np.square(noise)))


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int((time.time() - self.start_time)*1000)


def initialize_config(module_cfg, pass_args=True):
    """According to config items, load specific module dynamically with params.
    e.g., Config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])


def compute_PESQ(clean_signal, noisy_signal, sr=8000):
    try:
        pesq_out = pesq(sr, clean_signal, noisy_signal, "nb")
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("An exception occurred:", exc_type)
        print(exc_value)
        pesq_out = 0
    return pesq_out


def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_length

    return data_a[start:end], data_b[start:end]
    

def sample_fixed_length_data_aligned_rotor(data_a, data_b, sample_length, rotor):
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_length

    '''
    rotors_number = rotor.shape[1]
    rotor_data = np.random.rand(rotors_number, frames_total)
    for j in range(rotors_number):
        rotor_data[j] = sg.resample(rotor[:, j], frames_total)
    '''
    
    return data_a[start:end], data_b[start:end], rotor[:, start:end]


def compute_STOI(clean_signal, noisy_signal, sr=8000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)

def compute_SISDR(clean_signal, noisy_signal):
    si_sdr = ScaleInvariantSignalDistortionRatio()
    return si_sdr(torch.tensor(clean_signal), torch.tensor(noisy_signal))


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")


def compute_ESTOI(x, y, sr=8000):
    # Check if x and y have the same length
    if len(x) != len(y):
        raise ValueError('x and y should have the same length')
    
    # initialization
    x = x.reshape(-1, 1)  # clean speech column vector
    y = y.reshape(-1, 1)  # processed speech column vector
    
    fs = 8000  # sample rate of proposed intelligibility measure
    N_frame = 256  # window support
    K = 512  # FFT size
    J = 15  # Number of 1/3 octave bands
    mn = 150  # Center frequency of first 1/3 octave band in Hz.
    H, fc_thirdoct = thirdoct(fs, K, J, mn)  # Get 1/3 octave band matrix
    N = 30  # Number of frames for intermediate intelligibility measure
    dyn_range = 40  # speech dynamic range
    
    # resample signals if other samplerate is used than fs
    if sr != fs:
        x = sg.resample(x, int(len(x) * fs / sr))
        y = sg.resample(y, int(len(y) * fs / sr))
    
    # remove silent frames
    x, y = remove_silent_frames(x, y, dyn_range, N_frame, N_frame // 2)
    
    # apply 1/3 octave band TF-decomposition
    x_hat = stdft(x, N_frame, N_frame // 2, K)  # apply short-time DFT to clean speech
    y_hat = stdft(y, N_frame, N_frame // 2, K)  # apply short-time DFT to processed speech
    
    x_hat = x_hat[:, :K // 2 + 1].T  # take clean single-sided spectrum
    y_hat = y_hat[:, :K // 2 + 1].T  # take processed single-sided spectrum
    
    X = np.zeros((J, x_hat.shape[1]))  # init memory for clean speech 1/3 octave band TF-representation
    Y = np.zeros((J, y_hat.shape[1]))  # init memory for processed speech 1/3 octave band TF-representation
    
    for i in range(x_hat.shape[1]):
        X[:, i] = np.sqrt(H @ np.abs(x_hat[:, i]) ** 2)  # apply 1/3 octave band filtering
        Y[:, i] = np.sqrt(H @ np.abs(y_hat[:, i]) ** 2)
    
    # loop all segments of length N and obtain intermediate intelligibility measure for each
    d1 = np.zeros(len(range(N, X.shape[1])))

    for m in range(N, X.shape[1]):
        X_seg = X[:, m - N:m]  # region of length N with clean TF-units for all j
        Y_seg = Y[:, m - N:m]  # region of length N with processed TF-units for all j
        X_seg = X_seg + np.finfo(float).eps * np.random.randn(*X_seg.shape)  # to avoid divide by zero
        Y_seg = Y_seg + np.finfo(float).eps * np.random.randn(*Y_seg.shape)  # to avoid divide by zero

        # first normalize rows (to give \bar{S}_m)
        XX = X_seg - np.mean(X_seg, axis=1, keepdims=True)  # normalize rows to zero mean
        YY = Y_seg - np.mean(Y_seg, axis=1, keepdims=True)  # normalize rows to zero mean
        
        YY /= np.linalg.norm(YY)
        XX /= np.linalg.norm(XX)
        #YY = YY * np.diag(1. / np.sqrt(np.diag(YY @ YY.T)))  # normalize rows to unit length
        #XX = XX * np.diag(1. / np.sqrt(np.diag(XX @ XX.T)))  # normalize rows to unit length

        XX = XX + np.finfo(float).eps * np.random.randn(*XX.shape)  # to avoid corr.div.by.0
        YY = YY + np.finfo(float).eps * np.random.randn(*YY.shape)  # to avoid corr.div.by.0

        # then normalize columns (to give \check{S}_m)
        YYY = YY - np.mean(YY, axis=0, keepdims=True)  # normalize cols to zero mean
        XXX = XX - np.mean(XX, axis=0, keepdims=True)  # normalize cols to zero mean

        YYY = YYY @ np.diag(1. / np.sqrt(np.diag(YYY.T @ YYY)))  # normalize cols to unit length
        XXX = XXX @ np.diag(1. / np.sqrt(np.diag(XXX.T @ XXX)))  # normalize cols to unit length

        # compute average of col.correlations (by stacking cols)
        d1[m - N] = (1 / N) * (XXX.flatten() @ YYY.flatten())

    d = np.mean(d1)
    return d


def remove_silent_frames(x, y, range_val, N, K):
    x = x.squeeze()
    y = y.squeeze()
    
    frames = np.arange(0, len(x) - N + 1, K)
    w = np.hanning(N)
    msk = np.zeros(len(frames))
    
    for j in range(len(frames)):
        jj = np.arange(frames[j], frames[j] + N)
        msk[j] = 20 * np.log10(np.linalg.norm(x[jj] * w) / np.sqrt(N))
    
    msk = (msk - np.max(msk) + range_val) > 0
    count = 0
    
    x_sil = np.zeros_like(x)
    y_sil = np.zeros_like(y)
    
    for j in range(len(frames)):
        if msk[j]:
            jj_i = np.arange(frames[j], frames[j] + N)
            jj_o = np.arange(frames[count], frames[count] + N)
            x_sil[jj_o] += x[jj_i] * w
            y_sil[jj_o] += y[jj_i] * w
            count += 1
    
    x_sil = x_sil[:jj_o[-1] + 1]
    y_sil = y_sil[:jj_o[-1] + 1]
    
    return x_sil, y_sil


def thirdoct(fs, N_fft, numBands, mn):
    f = np.linspace(0, fs, N_fft + 1)
    f = f[:int(N_fft/2 + 1)]
    k = np.arange(numBands)
    cf = 2**(k/3) * mn
    fl = np.sqrt((2**(k/3) * mn) * (2**((k-1)/3) * mn))
    fr = np.sqrt((2**(k/3) * mn) * (2**((k+1)/3) * mn))
    A = np.zeros((numBands, len(f)))
    
    for i in range(len(cf)):
        fl_i = np.argmin((f - fl[i])**2)
        fl[i] = f[fl_i]
        
        fr_i = np.argmin((f - fr[i])**2)
        fr[i] = f[fr_i]
        
        A[i, fl_i:fr_i] = 1
    
    rnk = np.sum(A, axis=1)
    numBands = np.argmax((rnk[1:] >= rnk[:-1]) & (rnk[1:] != 0)) + 1
    A = A[:numBands, :]
    cf = cf[:numBands]
    
    return A, cf


def stdft(x, N, K, N_fft):
    frames = np.arange(0, len(x) - N + 1, K)
    x_stdft = np.zeros((len(frames), N_fft), dtype=np.complex128)

    w = np.hanning(N)
    x = x.flatten()

    for i, frame in enumerate(frames):
        ii = np.arange(frame, frame + N)
        x_stdft[i, :] = np.fft.fft(x[ii] * w, N_fft)

    return x_stdft


def binary_encode(x, max_value):
    if isinstance(x, int):
        x = np.array([x])
    if isinstance(x, list):
        x = np.array(x)
    width = np.ceil(np.log2(max_value)).astype(int)
    # 1 << [a, b, c] = [2^a, 2^b, 2^c].
    # & -> bitwise AND. > 0 -> Boolean. astype(int) -> 0,1.
    return (((x[:, None] & (1 << np.arange(width)))) > 0).astype(int)


def find_closest_value(array, target):
    abs_diff = np.abs(array - target)
    index = np.argmin(abs_diff)
    return index

def emphasis(signal_batch, emph_coeff=0.95, pre=True):
    """
    Pre-emphasis or De-emphasis of higher frequencies given a batch of signal.

    Args:
        signal_batch: batch of signals, represented as numpy arrays
        emph_coeff: emphasis coefficient
        pre: pre-emphasis or de-emphasis signals

    Returns:
        result: pre-emphasized or de-emphasized signal batch
    """
    result = np.zeros(signal_batch.shape)
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            if pre:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] - emph_coeff * channel_data[:-1])
            else:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] + emph_coeff * channel_data[:-1])
    return result


class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_type):

        if data_type == 'train':
            data_path = serialized_train_folder
        else:
            data_path = serialized_test_folder
        if not os.path.exists(data_path):
            raise FileNotFoundError('The {} data folder does not exist!'.format(data_type))

        self.data_type = data_type
        self.file_names = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]

    def reference_batch(self, batch_size):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.

        Args:
            batch_size(int): batch size

        Returns:
            ref_batch: reference batch
        """
        ref_file_names = np.random.choice(self.file_names, batch_size)
        ref_batch = np.stack([np.load(f) for f in ref_file_names])

        ref_batch = emphasis(ref_batch, emph_coeff=0.95)
        return torch.from_numpy(ref_batch).type(torch.FloatTensor)

    def __getitem__(self, idx):
        pair = np.load(self.file_names[idx])
        pair = emphasis(pair[np.newaxis, :, :], emph_coeff=0.95).reshape(2, -1)
        noisy = pair[1].reshape(1, -1)
        if self.data_type == 'train':
            clean = pair[0].reshape(1, -1)
            return torch.from_numpy(pair).type(torch.FloatTensor), torch.from_numpy(clean).type(
                torch.FloatTensor), torch.from_numpy(noisy).type(torch.FloatTensor)
        else:
            return os.path.basename(self.file_names[idx]), torch.from_numpy(noisy).type(torch.FloatTensor)

    def __len__(self):
        return len(self.file_names)
