import argparse
import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.autograd import Variable
from tqdm import tqdm
from soundpy import filtersignal

from segan_data_preprocess import slice_signal, window_size, sample_rate
from segan_model import Generator
from util.utils import emphasis, compute_ESTOI, compute_PESQ, compute_SISDR, compute_STOI, snr_db, ExecutionTime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Audio Enhancement')
    parser.add_argument('--file_name', type=str, required=True, help='directory audio name')
    parser.add_argument('--epoch_name', type=str, required=True, help='generator epoch name')

    opt = parser.parse_args()
    DIR_NAME = opt.file_name
    clean_dir = DIR_NAME.replace("noisy", "clean")
    enhanced_dir = DIR_NAME.replace("noisy", "enhanced")
    print(enhanced_dir)
    EPOCH_NAME = opt.epoch_name

    generator = Generator()
    generator.load_state_dict(torch.load('epochs/' + EPOCH_NAME, map_location='cpu'))
    if torch.cuda.is_available():
        generator.cuda()

    snr_pre = []
    snr_w = []
    snr_post = []
    estoi_pre = []
    estoi_w = []
    estoi_post = []
    pesq_pre = []
    pesq_w = []
    pesq_post = []
    stoi_pre = []
    stoi_w = []
    stoi_post = []
    sisdr_pre = []
    sisdr_w = []
    sisdr_post = []
    times_w = []
    times_segan = []

    for num, audio_file in enumerate(os.listdir(DIR_NAME)):
        FILE_NAME = os.path.join(DIR_NAME, audio_file)
        clean_file = os.path.join(clean_dir, audio_file)
        clean, _ = librosa.load(clean_file, sr=sample_rate)
        wav, noisy_slices = slice_signal(FILE_NAME, window_size, 1, sample_rate)
        timer_w = ExecutionTime()
        s_wf, _  = filtersignal(wav, sr=sample_rate, filter_type = 'wiener')
        times_w.append(timer_w.duration())
        snr_pre.append(snr_db(clean, wav))
        snr_w.append(snr_db(clean[:s_wf.shape[0]], s_wf))
        estoi_pre.append(compute_ESTOI(clean, wav))
        estoi_w.append(compute_ESTOI(clean[:s_wf.shape[0]], s_wf))
        pesq_pre.append(compute_PESQ(clean, wav))
        pesq_w.append(compute_PESQ(clean[:s_wf.shape[0]], s_wf))
        sisdr_pre.append(compute_SISDR(clean, wav))
        sisdr_w.append(compute_SISDR(clean[:s_wf.shape[0]], s_wf))
        stoi_pre.append(compute_STOI(clean, wav))
        stoi_w.append(compute_STOI(clean[:s_wf.shape[0]], s_wf))

        enhanced_speech = []
        timer_segan = ExecutionTime()
        for noisy_slice in tqdm(noisy_slices, desc='Generate enhanced audio'):
            z = nn.init.normal_(torch.Tensor(1, 1024, 8))
            noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
            if torch.cuda.is_available():
                noisy_slice, z = noisy_slice.cuda(), z.cuda()
            noisy_slice, z = Variable(noisy_slice), Variable(z)
            generated_speech = generator(noisy_slice, z).data.cpu().numpy()
            generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
            generated_speech = generated_speech.reshape(-1)
            enhanced_speech.append(generated_speech)
        times_segan.append(timer_segan.duration())
        enhanced_speech = np.array(enhanced_speech).reshape(1, -1)
        file_name = os.path.join(enhanced_dir,
                                'enhanced_{}.wav'.format(os.path.basename(FILE_NAME).split('.')[0]))
        wavfile.write(file_name, sample_rate, enhanced_speech.T)
        enhanced = np.squeeze(enhanced_speech.T)
        print(enhanced.shape)
        snr_post.append(snr_db(clean, enhanced))
        estoi_post.append(compute_ESTOI(clean, enhanced))
        pesq_post.append(compute_PESQ(clean, enhanced))
        sisdr_post.append(compute_SISDR(clean, enhanced))
        stoi_post.append(compute_STOI(clean, enhanced))
        if num == 10:
            break
    print('MEAN SNR PRE', np.mean(snr_pre))
    print('SD SNR PRE', np.std(snr_pre))
    print('MEAN SNR WIENER', np.mean(snr_w))
    print('SD SNR WIENER', np.std(snr_w))
    print('MEAN SNR SEGAN', np.mean(snr_post))
    print('SD SNR SEGAN', np.std(snr_post))

    print('MEAN SISDR PRE', np.mean(sisdr_pre))
    print('SD SISDR PRE', np.std(sisdr_pre))
    print('MEAN SISDR WIENER', np.mean(sisdr_w))
    print('SD SISDR WIENER', np.std(sisdr_w))
    print('MEAN SISDR SEGAN', np.mean(sisdr_post))
    print('SD SISDR SEGAN', np.std(sisdr_post))

    print('MEAN STOI PRE', np.mean(stoi_pre))
    print('SD STOI PRE', np.std(stoi_pre))
    print('MEAN STOI WIENER', np.mean(stoi_w))
    print('SD STOI WIENER', np.std(stoi_w))
    print('MEAN STOI SEGAN', np.mean(stoi_post))
    print('SD STOI SEGAN', np.std(stoi_post))

    print('MEAN ESTOI PRE', np.mean(estoi_pre))
    print('SD ESTOI PRE', np.std(estoi_pre))
    print('MEAN ESTOI WIENER', np.mean(estoi_w))
    print('SD ESTOI WIENER', np.std(estoi_w))
    print('MEAN ESTOI SEGAN', np.mean(estoi_post))
    print('SD ESTOI SEGAN', np.std(estoi_post))

    print('MEAN PESQ PRE', np.mean(pesq_pre))
    print('SD PESQ PRE', np.std(pesq_pre))
    print('MEAN PESQ WIENER', np.mean(pesq_w))
    print('SD PESQ WIENER', np.std(pesq_w))
    print('MEAN PESQ SEGAN', np.mean(pesq_post))
    print('SD PESQ SEGAN', np.std(pesq_post))

    print('MEAN TIME WIENER', np.mean(times_w))
    print('SD TIME WIENER', np.std(times_w))
    print('MEAN TIME SEGAN', np.mean(times_segan))
    print('SD TIME SEGAN', np.std(times_segan))

