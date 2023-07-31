import argparse
import json
import json5
import os

import soundfile as sf
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from util.utils import initialize_config, load_checkpoint, compute_ESTOI, compute_PESQ, compute_SISDR, compute_STOI, snr_db, ExecutionTime
from model.unet_basic import Model as wunet_basic
from dataset_creation import DatasetCreator
from dataset.waveform_dataset import Dataset as loader


from soundpy import filtersignal


def main(args):
    """
    Preparation
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = json.load(open(args.config))
    model_checkpoint_path = args.model_checkpoint_path
    output_dir = args.output_dir
    assert os.path.exists(output_dir), "Enhanced directory should be exist."

    """
    DataLoader
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    current_dir = os.getcwd() #TODO!
    config_dataset = json5.load(open(os.path.join(current_dir, 'config/dataset_creation/dataset_create.json')))
    datesetCreator = DatasetCreator(current_dir, config_dataset)
    if args.snr != 'random':
        datesetCreator.test_at_SNR_no_rotors(float(args.snr))
    test_dataset = initialize_config(config["dataset"])
    dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    #dataloader = DataLoader(dataset=initialize_config(config["dataset"]), batch_size=1, num_workers=0)
    """
    Model
    """
    model = initialize_config(config["model"])
    model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
    model.to(device)
    model.eval()

    """
    Enhancement
    """
    sample_length = config["custom"]["sample_length"]
    sample_rate = config["custom"]["sample_rate"]

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
    times_wun = []

    for (mixture, clean, name) in tqdm(dataloader):
        assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
        name = name[0]
        padded_length = 0
        no_model_mixture = mixture.cpu().numpy()
        no_model_mixture = np.squeeze(no_model_mixture)
        no_model_clean = clean.cpu().numpy()
        no_model_clean = np.squeeze(no_model_clean)
        timer_w = ExecutionTime()
        s_wf, _  = filtersignal(no_model_mixture, sr=sample_rate, filter_type = 'wiener')
        times_w.append(timer_w.duration())
        snr_pre.append(snr_db(no_model_clean, no_model_mixture))
        snr_w.append(snr_db(no_model_clean[:s_wf.shape[0]], s_wf))
        estoi_pre.append(compute_ESTOI(no_model_clean, no_model_mixture))
        estoi_w.append(compute_ESTOI(no_model_clean[:s_wf.shape[0]], s_wf))
        pesq_pre.append(compute_PESQ(no_model_clean, no_model_mixture))
        pesq_w.append(compute_PESQ(no_model_clean[:s_wf.shape[0]], s_wf))
        sisdr_pre.append(compute_SISDR(no_model_clean, no_model_mixture))
        sisdr_w.append(compute_SISDR(no_model_clean[:s_wf.shape[0]], s_wf))
        stoi_pre.append(compute_STOI(no_model_clean, no_model_mixture))
        stoi_w.append(compute_STOI(no_model_clean[:s_wf.shape[0]], s_wf))
        mixture = mixture.to(device)  # [1, 1, T]
        clean = clean.to(device)  # [1, 1, T]

        # The input of the model should be fixed length.
        if mixture.size(-1) % sample_length != 0:
            padded_length = sample_length - (mixture.size(-1) % sample_length)
            mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=device)], dim=-1)

        if clean.size(-1) % sample_length != 0:
            padded_length = sample_length - (clean.size(-1) % sample_length)
            clean = torch.cat([clean, torch.zeros(1, 1, padded_length, device=device)], dim=-1)
        
        assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
        mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))
        
        enhanced_chunks = []
        timer_wun = ExecutionTime()
        for chunk in mixture_chunks:
            enhanced_chunks.append(model(chunk).detach().cpu())
        times_wun.append(timer_wun.duration())

        enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
        enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]

        enhanced = enhanced.reshape(-1).numpy()

        output_path = os.path.join(output_dir, f"{name}.wav")
        sf.write(output_path, enhanced, sample_rate)

        clean = clean.reshape(-1).numpy()
        snr_post.append(snr_db(clean, enhanced))
        estoi_post.append(compute_ESTOI(clean, enhanced))
        pesq_post.append(compute_PESQ(clean, enhanced))
        sisdr_post.append(compute_SISDR(clean, enhanced))
        stoi_post.append(compute_STOI(clean, enhanced))
        output_path = os.path.join(output_dir, f"{name}.wav")
        sf.write(output_path, enhanced, sample_rate)
    print('MEAN SNR PRE', np.mean(snr_pre))
    print('SD SNR PRE', np.std(snr_pre))
    print('MEAN SNR WIENER', np.mean(snr_w))
    print('SD SNR WIENER', np.std(snr_w))
    print('MEAN SNR WAVE-U-NET', np.mean(snr_post))
    print('SD SNR WAVE-U-NET', np.std(snr_post))

    print('MEAN SISDR PRE', np.mean(sisdr_pre))
    print('SD SISDR PRE', np.std(sisdr_pre))
    print('MEAN SISDR WIENER', np.mean(sisdr_w))
    print('SD SISDR WIENER', np.std(sisdr_w))
    print('MEAN SISDR WAVE-U-NET', np.mean(sisdr_post))
    print('SD SISDR WAVE-U-NET', np.std(sisdr_post))

    print('MEAN STOI PRE', np.mean(stoi_pre))
    print('SD STOI PRE', np.std(stoi_pre))
    print('MEAN STOI WIENER', np.mean(stoi_w))
    print('SD STOI WIENER', np.std(stoi_w))
    print('MEAN STOI WAVE-U-NET', np.mean(stoi_post))
    print('SD STOI WAVE-U-NET', np.std(stoi_post))

    print('MEAN ESTOI PRE', np.mean(estoi_pre))
    print('SD ESTOI PRE', np.std(estoi_pre))
    print('MEAN ESTOI WIENER', np.mean(estoi_w))
    print('SD ESTOI WIENER', np.std(estoi_w))
    print('MEAN ESTOI WAVE-U-NET', np.mean(estoi_post))
    print('SD ESTOI WAVE-U-NET', np.std(estoi_post))

    print('MEAN PESQ PRE', np.mean(pesq_pre))
    print('SD PESQ PRE', np.std(pesq_pre))
    print('MEAN PESQ WIENER', np.mean(pesq_w))
    print('SD PESQ WIENER', np.std(pesq_w))
    print('MEAN PESQ WAVE-U-NET', np.mean(pesq_post))
    print('SD PESQ WAVE-U-NET', np.std(pesq_post))

    print('MEAN TIME WIENER', np.mean(times_w))
    print('SD TIME WIENER', np.std(times_w))
    print('MEAN TIME WAVE-U-NET', np.mean(times_wun))
    print('SD TIME WAVE-U-NET', np.std(times_wun))

if __name__ == '__main__':
    """
    Parameters
    """
    parser = argparse.ArgumentParser("Wave-U-Net: Speech Enhancement")
    parser.add_argument("-C", "--config", type=str, required=True, help="Model and dataset for enhancement (*.json).")
    parser.add_argument("-D", "--device", default="-1", type=str, help="GPU for speech enhancement. default: CPU")
    parser.add_argument("-O", "--output_dir", type=str, required=True, help="Where are audio save.")
    parser.add_argument("-S", "--snr", type=str, required=True, help="SNR of the test set mixture")
    parser.add_argument("-M", "--model_checkpoint_path", type=str, required=True, help="Checkpoint.")
    args = parser.parse_args()

    main(args)


def enhance(mixture, model_checkpoint_path, sample_length):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = wunet_basic()
    model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
    model.to(device)
    model.eval()

    padded_length = 0
    mixture = torch.from_numpy(np.reshape(mixture, (1, 1, -1)))
    mixture = mixture.to(device, dtype=torch.float)  # [1, 1, T]

    # The input of the model should be fixed length.
    if mixture.size(-1) % sample_length != 0:
        padded_length = sample_length - (mixture.size(-1) % sample_length)
        mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=device)], dim=-1)
    
    assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
    mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))
    
    enhanced_chunks = []
    for chunk in mixture_chunks:
        enhanced_chunks.append(model(chunk).detach().cpu())

    enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
    enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]

    enhanced = enhanced.reshape(-1).numpy()

    return enhanced