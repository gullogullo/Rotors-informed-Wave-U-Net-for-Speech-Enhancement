import os
import json5
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.base_trainer import BaseTrainer
from dataset.waveform_dataset import Dataset as loader
from dataset.waveform_dataset_rotors import Dataset as loader_rotors
from dataset_creation import DatasetCreator
from util.utils import compute_STOI, compute_PESQ, compute_ESTOI, compute_SISDR, binary_encode
plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.validation_data_loader = validation_dataloader
        #current_dir = os.getcwd() #TODO!
        #config = json5.load(open(os.path.join(current_dir, 'config/dataset_creation/dataset_create.json')))
        #self.datesetCreator = DatasetCreator(current_dir, config)
        #print('generating training data...')
        self.train_dataloader = train_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0
        for n in range(1): #10
            #print('generating training data...')
            #train_dataloader = DataLoader(loader("Datasets/train_dataset_rotors.txt",  limit=None, 
            #                                     offset=0, mode="train", sample_length=16384, num_speakers=1))
            for i, (mixture, clean, name) in enumerate(self.train_dataloader):
                mixture = mixture.to(self.device)
                clean = clean.to(self.device)

                self.optimizer.zero_grad()
                enhanced = self.model(mixture)
                loss = self.loss_function(clean, enhanced)
                if loss > 1:
                    print('train loss single batch', loss)
                loss.backward()
                self.optimizer.step()

                loss_total += loss.item()

        dl_len = len(self.train_dataloader) * 10
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)
        return loss_total / dl_len

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []
        estoi_c_n = []
        estoi_c_e = []
        sisdr_c_n = []
        sisdr_c_e = []

        loss_total = 0
        for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            mixture = mixture.to(self.device)  # [1, 1, T]

            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)

            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            enhanced_chunks = []
            for chunk in mixture_chunks:
                enhanced_chunks.append(self.model(chunk).detach().cpu())

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            mixture = mixture if padded_length == 0 else mixture[:, :, :-padded_length]

            loss = self.loss_function(clean, enhanced)
            #print('valid loss single batch', loss)
            loss_total += loss.item()

            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=8000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveshow(y, sr=8000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
            noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    noisy_mag,
                    enhanced_mag,
                    clean_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=8000)
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metric
            stoi_c_n.append(compute_STOI(clean, mixture, sr=8000))
            stoi_c_e.append(compute_STOI(clean, enhanced, sr=8000))
            pesq = compute_PESQ(clean, mixture, sr=8000)
            if pesq != 0:
                pesq_c_n.append(pesq)
            pesq = compute_PESQ(clean, enhanced, sr=8000)
            if pesq != 0:
                pesq_c_e.append(pesq)
            #pesq_c_n.append(compute_PESQ(clean, mixture, sr=8000))
            #pesq_c_e.append(compute_PESQ(clean, enhanced, sr=8000))
            estoi_c_n.append(compute_ESTOI(clean, mixture, sr=8000))
            estoi_c_e.append(compute_ESTOI(clean, enhanced, sr=8000))
            sisdr_c_n.append(compute_SISDR(clean, mixture))
            sisdr_c_e.append(compute_SISDR(clean, enhanced))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metric/STOI", {
            "Clean and noisy": get_metrics_ave(stoi_c_n),
            "Clean and enhanced": get_metrics_ave(stoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/PESQ", {
            "Clean and noisy": get_metrics_ave(pesq_c_n),
            "Clean and enhanced": get_metrics_ave(pesq_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/ESTOI", {
            "Clean and noisy": get_metrics_ave(estoi_c_n),
            "Clean and enhanced": get_metrics_ave(estoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/SISDR", {
            "Clean and noisy": get_metrics_ave(sisdr_c_n),
            "Clean and enhanced": get_metrics_ave(sisdr_c_e)
        }, epoch)

        score = (get_metrics_ave(stoi_c_e) + self._transform_pesq_range(get_metrics_ave(pesq_c_e))) / 2
        dl_len = len(self.validation_data_loader)
        return score, loss_total / dl_len


class RotorTrainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader
    ):
        super(RotorTrainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.validation_data_loader = validation_dataloader
        #current_dir = os.getcwd() #TODO!
        #config = json5.load(open(os.path.join(current_dir, 'config/dataset_creation/dataset_create.json')))
        #self.datesetCreator = DatasetCreator(current_dir, config)
        #print('generating training data...')
        self.train_dataloader = train_dataloader


    def _train_epoch(self, epoch):
        loss_total = 0.0
        for n in range(1): #10
            #print('generating training data...')
            #train_dataloader = DataLoader(loader_rotors("Datasets/train_dataset_rotors.txt", limit=None, 
            #                                            offset=0, mode="train", sample_length=16384, num_speakers=1))
            for i, (mixture, clean, rotor, name) in enumerate(self.train_dataloader):
                mixture = mixture.to(self.device)
                clean = clean.to(self.device)
                rotor = rotor.to(self.device, dtype=torch.float)
                self.optimizer.zero_grad()
                enhanced = self.model(mixture, rotor)
                loss = self.loss_function(clean, enhanced)
                if loss > 1:
                    print('train loss single batch', loss)
                loss.backward()
                self.optimizer.step()

                loss_total += loss.item()

        dl_len = len(self.train_dataloader) * 10
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)
        return loss_total / dl_len

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []
        estoi_c_n = []
        estoi_c_e = []
        sisdr_c_n = []
        sisdr_c_e = []

        loss_total = 0
        for i, (mixture, clean, rotor, name) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            mixture = mixture.to(self.device)  # [1, 1, T]
            rotor = rotor.to(self.device, dtype=torch.float)
            #print(rotor.size())
            #rotor = torch.permute(rotor, (0, 2, 1))
            #print(rotor.size())

            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)
            
            if rotor.size(-1) % sample_length != 0:
                padded_length_rotor = sample_length - (rotor.size(-1) % sample_length) 
                rotor = torch.cat([rotor, torch.zeros(1, 4, padded_length_rotor, device=self.device)], dim=-1)

            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            assert rotor.size(-1) % sample_length == 0 and rotor.dim() == 3
            rotor_chunks = list(torch.split(rotor, sample_length, dim=-1))
            enhanced_chunks = []
            for n, chunk in enumerate(mixture_chunks):
                enhanced_chunks.append(self.model(chunk, rotor_chunks[n]).detach().cpu())

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            mixture = mixture if padded_length == 0 else mixture[:, :, :-padded_length]

            loss = self.loss_function(clean, enhanced)
            #print('valid loss single batch', loss)
            loss_total += loss.item()

            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=8000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveshow(y, sr=8000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
            noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    noisy_mag,
                    enhanced_mag,
                    clean_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=8000)
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metric
            #print('name', name)
            #print('clean', clean.shape)
            #print('mixture', mixture.shape)
            #print('enhanced', enhanced.shape)
            stoi_c_n.append(compute_STOI(clean, mixture, sr=8000))
            stoi_c_e.append(compute_STOI(clean, enhanced, sr=8000))
            pesq = compute_PESQ(clean, mixture, sr=8000)
            if pesq != 0:
                pesq_c_n.append(pesq)
            pesq = compute_PESQ(clean, enhanced, sr=8000)
            if pesq != 0:
                pesq_c_e.append(pesq)
            estoi_c_n.append(compute_ESTOI(clean, mixture, sr=8000))
            estoi_c_e.append(compute_ESTOI(clean, enhanced, sr=8000))
            sisdr_c_n.append(compute_SISDR(clean, mixture))
            sisdr_c_e.append(compute_SISDR(clean, enhanced))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metric/STOI", {
            "Clean and noisy": get_metrics_ave(stoi_c_n),
            "Clean and enhanced": get_metrics_ave(stoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/PESQ", {
            "Clean and noisy": get_metrics_ave(pesq_c_n),
            "Clean and enhanced": get_metrics_ave(pesq_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/ESTOI", {
            "Clean and noisy": get_metrics_ave(estoi_c_n),
            "Clean and enhanced": get_metrics_ave(estoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/SISDR", {
            "Clean and noisy": get_metrics_ave(sisdr_c_n),
            "Clean and enhanced": get_metrics_ave(sisdr_c_e)
        }, epoch)

        dl_len = len(self.validation_data_loader)
        score = (get_metrics_ave(stoi_c_e) + self._transform_pesq_range(get_metrics_ave(pesq_c_e))) / 2
        return score, loss_total / dl_len
    


# THE FOLLOWING ARE STILL IN DEVELOPMENT
class RotorIdTrainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(RotorIdTrainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        self.num_speakers = config["model"]["args"]["num_speakers"]
        self.speakers_set = []
        self.config = config

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for i, (mixture, clean, rotor, name) in enumerate(self.train_data_loader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)
            rotor = rotor.to(self.device, dtype=torch.float)

            if np.random.uniform(0, 1) <= 1.0 / self.num_speakers:
                    speaker_id = torch.tensor(0)
            else:
                speaker_id = torch.tensor(binary_encode(name, int(self.config["model"]["args"]["num_speakers"]) + 1))
                self.speakers_set.append(speaker_id)
            #print('name', name)
            speaker_id = speaker_id.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()
            enhanced = self.model(mixture, rotor, speaker_id)
            loss = self.loss_function(clean, enhanced)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []
        estoi_c_n = []
        estoi_c_e = []
        sisdr_c_n = []
        sisdr_c_e = []


        for i, (mixture, clean, rotor, name) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            mixture = mixture.to(self.device)  # [1, 1, T]
            rotor = rotor.to(self.device, dtype=torch.float)
            rotor = torch.permute(rotor, (0, 2, 1))

            speaker_id = torch.tensor(hash(name))
            #print('name', name)
            if speaker_id not in self.speakers_set:
                speaker_id = torch.tensor(0)
            speaker_id = speaker_id.to(self.device, dtype=torch.float)

            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)
            
            if rotor.size(-1) % sample_length != 0:
                padded_length_rotor = sample_length - (rotor.size(-1) % sample_length) 
                rotor = torch.cat([rotor, torch.zeros(1, 4, padded_length_rotor, device=self.device)], dim=-1)

            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            assert rotor.size(-1) % sample_length == 0 and rotor.dim() == 3
            rotor_chunks = list(torch.split(rotor, sample_length, dim=-1))
            enhanced_chunks = []
            for n, chunk in enumerate(mixture_chunks):
                enhanced_chunks.append(self.model(chunk, rotor_chunks[n], speaker_id).detach().cpu())

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            mixture = mixture if padded_length == 0 else mixture[:, :, :-padded_length]

            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=8000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveshow(y, sr=8000, ax=ax[j])
                plt.tight_layout()
                #print('doing the waveform plot')
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
            noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    noisy_mag,
                    enhanced_mag,
                    clean_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=8000)
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metric
            stoi_c_n.append(compute_STOI(clean, mixture, sr=8000))
            stoi_c_e.append(compute_STOI(clean, enhanced, sr=8000))
            pesq_c_n.append(compute_PESQ(clean, mixture, sr=8000))
            pesq_c_e.append(compute_PESQ(clean, enhanced, sr=8000))
            estoi_c_n.append(compute_ESTOI(clean, mixture, sr=8000))
            estoi_c_e.append(compute_ESTOI(clean, enhanced, sr=8000))
            sisdr_c_n.append(compute_SISDR(clean, mixture))
            sisdr_c_e.append(compute_SISDR(clean, enhanced))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metric/STOI", {
            "Clean and noisy": get_metrics_ave(stoi_c_n),
            "Clean and enhanced": get_metrics_ave(stoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/PESQ", {
            "Clean and noisy": get_metrics_ave(pesq_c_n),
            "Clean and enhanced": get_metrics_ave(pesq_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/ESTOI", {
            "Clean and noisy": get_metrics_ave(estoi_c_n),
            "Clean and enhanced": get_metrics_ave(estoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/SISDR", {
            "Clean and noisy": get_metrics_ave(sisdr_c_n),
            "Clean and enhanced": get_metrics_ave(sisdr_c_e)
        }, epoch)

        score = (get_metrics_ave(stoi_c_e) + self._transform_pesq_range(get_metrics_ave(pesq_c_e))) / 2
        return score
    

class WaveNetTrainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader
    ):
        super(WaveNetTrainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        self.num_speakers = config["model"]["args"]["num_speakers"]
        self.speakers_set = []
        self.config = config
        self.model = model
        self.batch_size = config["train_dataloader"]["batch_size"]
        self.input_length = config["model"]["args"]["input_length"]
        self.receptive_field_length = int(model.receptive_field_length)
        self.target_field_length = int(config["model"]["args"]["target_field_length"])
        self.num_fragments = int(np.floor(self.input_length / self.receptive_field_length))
        self.offset = self.model.half_receptive_field_length + int(self.target_field_length * 0.5)
        #self.offset = self.target_field_length
        self.binary_width = np.ceil(np.log2(self.config["model"]["args"]["num_speakers"] + 1)).astype(int)


    def _train_epoch(self, epoch):
        loss_total = 0.0
        for i, (mixture, clean, name) in enumerate(self.train_data_loader):
            speaker_ids = []
            for string_name in name:
                if string_name not in self.speakers_set:
                    speaker_ids.append([binary_encode(len(self.speakers_set) + 1, int(self.config["model"]["args"]["num_speakers"]) + 1)] * self.num_fragments)
                    self.speakers_set.append(string_name)
                else:
                    speaker_ids.append([binary_encode(self.speakers_set.index(string_name), int(self.config["model"]["args"]["num_speakers"]) + 1)] * self.num_fragments)
                #if np.random.uniform(0, 1) <= 1.0 / self.num_speakers:
                #        speaker_id = torch.tensor(0)
                #else:
                #    speaker_id = torch.tensor(binary_encode(string_name, self.config["model"]["args"]["num_speakers"]))
                #    self.speakers_set.append(speaker_id)
            speaker_ids = np.asarray(speaker_ids)
            speaker_ids = speaker_ids.reshape(-1, 1, self.binary_width)
            #print('name', name)
            #print('speaker_ids', speaker_ids)
            #print('self.speakers_set', self.speakers_set)
            if mixture.size()[-1] < self.receptive_field_length:
                raise ValueError('Input is not long enough to be used with this model.')

            remainder = int(self.input_length) % self.target_field_length
            print('remainder', remainder)
            mixlist = torch.rand((self.num_fragments * self.batch_size, 1, self.receptive_field_length))
            cleanlist = torch.rand((self.num_fragments * self.batch_size, 1, self.model.padded_target_field_length))
            for n_mix, mix in enumerate(mixture):
                fragment_i = 0
                for n_fragment in range(self.num_fragments):
                    mixlist[(n_mix + 1) * n_fragment, :, :] = mix[:, slice(fragment_i, fragment_i + self.receptive_field_length, 1)]
                    cleanlist[(n_mix + 1) * n_fragment, :, :] = clean[n_mix, :, slice(self.offset, self.offset + self.model.padded_target_field_length, 1)]
                    fragment_i += self.receptive_field_length
            for fragment in range(self.num_fragments):
                mixture_batch = mixlist[self.batch_size * fragment: self.batch_size * (fragment + 1), :, :]
                clean_batch = cleanlist[self.batch_size * fragment: self.batch_size * (fragment + 1), :, :]
                speaker_batch = speaker_ids[self.batch_size * fragment: self.batch_size * (fragment + 1), :, :]
                speaker_batch = torch.tensor(speaker_batch)
                speaker_batch = speaker_batch.to(self.device, dtype=torch.float)
                mixture_batch = mixture_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)
                self.optimizer.zero_grad()
                enhanced, noise_estimate = self.model(mixture_batch, speaker_batch)
                #print('clean_slice', clean_batch.size())
                #print('enhanced', enhanced.size())
                loss = self.loss_function(clean_batch, enhanced)
                loss.backward()
                self.optimizer.step()

                loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []
        estoi_c_n = []
        estoi_c_e = []
        sisdr_c_n = []
        sisdr_c_e = []


        for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            mixture = mixture.to(self.device)  # [1, 1, T]
            # The input of the model should be fixed length.
            if mixture.size(-1) % self.receptive_field_length != 0:
                padded_length = self.receptive_field_length - (mixture.size(-1) % self.receptive_field_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)
            assert mixture.size(-1) % self.receptive_field_length == 0 and mixture.dim() == 3
            #mixture_chunks = list(torch.split(mixture, self.receptive_field_length, dim=-1))
            mixture_chunks = []
            #clean_cut = clean[:, :, slice(self.offset, self.offset + self.model.padded_target_field_length, 1)]
            num_fragments = int(np.floor(mixture.size(-1) / self.receptive_field_length))
            fragment_i = 0
            for n_fragment in range(num_fragments):
                mixture_chunks.append(mixture[:, :, slice(fragment_i, fragment_i + self.receptive_field_length, 1)])
                #clean_chunks.append(clean[:, :, slice(self.offset, self.offset + self.model.padded_target_field_length, 1)])
                fragment_i += self.receptive_field_length
                #fragment_i += self.offset


            if name not in self.speakers_set:
                speaker_id = torch.tensor([0] * self.binary_width)
            else:
                speaker_id = torch.tensor(binary_encode(self.speakers_set.index(name), int(self.config["model"]["args"]["num_speakers"]) + 1))
            #print('speaker_id', speaker_id) 
            speaker_id = speaker_id.unsqueeze_(1)
            #speaker_id = speaker_id.repeat(1, 1, len(mixture_chunks))
            #speaker_id = torch.reshape(speaker_id, (1, 1, len(mixture_chunks) * self.binary_width))
            speaker_id = speaker_id.to(self.device, dtype=torch.float)
            #print('name', name)
            #print('speaker_id', speaker_id)
            enhanced_chunks = []
            for n, chunk in enumerate(mixture_chunks):
                enhanced_chunk, _ = self.model(chunk, speaker_id)
                enhanced_chunks.append(enhanced_chunk.detach().cpu())

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            #enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            #mixture = mixture if padded_length == 0 else mixture[:, :, :-padded_length]
            mixture = mixture[:, :, self.offset:self.offset + enhanced.size(-1)]
            clean = clean[:, :, self.offset:self.offset + enhanced.size(-1)]
            #print('enhanced', enhanced.size())
            #print('clean', clean.size())
            #print('mixture', mixture.size())
            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=8000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveshow(y, sr=8000, ax=ax[j])
                plt.tight_layout()
                #print('doing the waveform plot')
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
            noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    noisy_mag,
                    enhanced_mag,
                    clean_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=8000)
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metric
            stoi_c_n.append(compute_STOI(clean, mixture, sr=8000))
            stoi_c_e.append(compute_STOI(clean, enhanced, sr=8000))
            pesq_c_n.append(compute_PESQ(clean, mixture, sr=8000))
            pesq_c_e.append(compute_PESQ(clean, enhanced, sr=8000))
            estoi_c_n.append(compute_ESTOI(clean, mixture, sr=8000))
            estoi_c_e.append(compute_ESTOI(clean, enhanced, sr=8000))
            sisdr_c_n.append(compute_SISDR(clean, mixture))
            sisdr_c_e.append(compute_SISDR(clean, enhanced))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metric/STOI", {
            "Clean and noisy": get_metrics_ave(stoi_c_n),
            "Clean and enhanced": get_metrics_ave(stoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/PESQ", {
            "Clean and noisy": get_metrics_ave(pesq_c_n),
            "Clean and enhanced": get_metrics_ave(pesq_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/ESTOI", {
            "Clean and noisy": get_metrics_ave(estoi_c_n),
            "Clean and enhanced": get_metrics_ave(estoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/SISDR", {
            "Clean and noisy": get_metrics_ave(sisdr_c_n),
            "Clean and enhanced": get_metrics_ave(sisdr_c_e)
        }, epoch)

        score = (get_metrics_ave(stoi_c_e) + self._transform_pesq_range(get_metrics_ave(pesq_c_e))) / 2
        return score