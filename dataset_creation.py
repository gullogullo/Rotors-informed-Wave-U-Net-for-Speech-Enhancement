import os
import random
import argparse
import json

import numpy as np

import scipy.signal as sps
from scipy.io import wavfile
from scipy.io import loadmat

import soundfile as sf

from util.utils import normalize, snr_db


def main(config, on_the_fly):

    current_dir = os.getcwd() #TODO!
    datesetCreator = DatasetCreator(current_dir, config)

    clean_dictionaries = datesetCreator.clean_speech_speaker_id(on_the_fly=on_the_fly)
    if on_the_fly == 'SEGAN':
        no_noise = 0
        no_bg_noise = 0
        no_egonoise = 0
        no_speech = 0
        single_rotor = 0

        clean_dictionaries = datesetCreator.clean_speech_speaker_id(on_the_fly=on_the_fly)
        train_dictionary = clean_dictionaries[0]
        clean_sequences = train_dictionary['clean_sequences']
        for file_name in os.listdir(datesetCreator.clean_train_path):
            file_path = os.path.join(datesetCreator.clean_train_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(datesetCreator.noisy_train_path):
            file_path = os.path.join(datesetCreator.noisy_train_path, file_name)
            os.remove(file_path)
        #for file_name in os.listdir(datesetCreator.rotor_train_path):
        #    file_path = os.path.join(datesetCreator.rotor_train_path, file_name)
        #    os.remove(file_path)
        with open(datesetCreator.txt_paths[0], 'w') as txtfile:
            for n, clean_sequence in enumerate(clean_sequences):
                wav_path = train_dictionary['clean_wav_filenames'][n].split('.')
                wav_filename = train_dictionary['speakers'][n] + '_' + wav_path[0]
                txtfile.write(str(os.path.join(datesetCreator.noisy_train_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(datesetCreator.clean_train_path, wav_filename + '.wav')) + ' ')
                #txtfile.write(str(os.path.join(datesetCreator.rotor_train_path, wav_filename + '.csv')) + '\n')
                if np.random.uniform(0, 1) <= datesetCreator.single_rps_percent:
                    single_rotor += 1
                    dregon_id = 0
                else:
                    dregon_id = 2
                if np.random.uniform(0, 1) <= datesetCreator.speech_only_percent:
                    no_noise += 1
                    wavfile.write(os.path.join(datesetCreator.clean_train_path, wav_filename + '.wav'), datesetCreator.sample_rate, clean_sequence)
                    wavfile.write(os.path.join(datesetCreator.noisy_train_path, wav_filename + '.wav'), datesetCreator.sample_rate, clean_sequence)
                else:
                    if np.random.uniform(0, 1) <= datesetCreator.pure_egonoise_percent:
                        no_bg_noise += 1
                        egonoise_dictionary = datesetCreator.egonoise_rotors(dregon_id=dregon_id)
                        noise_sequence = egonoise_dictionary['egonoise_sequence'][0]
                    elif np.random.uniform(0, 1) <= datesetCreator.no_rotor_percent:
                        no_egonoise += 1
                        bg_dictionary = datesetCreator.background_noise(partition='train')
                        noise_sequence = bg_dictionary['bg_noise_sequence'][0]
                    else:
                        egonoise_dictionary = datesetCreator.egonoise_rotors(dregon_id=dregon_id)
                        bg_dictionary = datesetCreator.background_noise(partition='train')
                        snr_noises = random.choice(datesetCreator.snr_noises)
                        noise_sequence, _, _, _, _ = datesetCreator.mix_snr(egonoise_dictionary['egonoise_sequence'][0], bg_dictionary['bg_noise_sequence'][0], snr_noises)
                    if np.random.uniform(0, 1) <= datesetCreator.noise_only_percent:
                        no_speech += 1
                        wavfile.write(os.path.join(datesetCreator.clean_train_path, wav_filename + '.wav'), datesetCreator.sample_rate, clean_sequence * 0)
                        wavfile.write(os.path.join(datesetCreator.noisy_train_path, wav_filename + '.wav'), datesetCreator.sample_rate, noise_sequence)
                    else:
                        snr_train = np.random.uniform(datesetCreator.snrs_train[0], datesetCreator.snrs_train[1])
                        mixture, signal, _, _, _ = datesetCreator.mix_snr(clean_sequence, noise_sequence, snr_train)
                        wavfile.write(os.path.join(datesetCreator.clean_train_path, wav_filename + '.wav'), datesetCreator.sample_rate, signal)
                        wavfile.write(os.path.join(datesetCreator.noisy_train_path, wav_filename + '.wav'), datesetCreator.sample_rate, mixture)

        print('TRAIN No noise ', no_noise / len(clean_sequences) * 100)
        print('TRAIN No background noise ', no_bg_noise / len(clean_sequences) * 100)
        print('TRAIN No egonoise ', no_egonoise / len(clean_sequences) * 100)
        print('TRAIN No speech ', no_speech / len(clean_sequences) * 100)
        print('TRAIN Single rotor ', single_rotor / len(clean_sequences) * 100)

        no_bg_noise = 0
        no_egonoise = 0
        no_speech = 0
        single_rotor = 0

        valid_dictionary = clean_dictionaries[0]
        clean_valid_sequences = valid_dictionary['clean_sequences']
        for file_name in os.listdir(datesetCreator.clean_valid_path):
            file_path = os.path.join(datesetCreator.clean_valid_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(datesetCreator.noisy_valid_path):
            file_path = os.path.join(datesetCreator.noisy_valid_path, file_name)
            os.remove(file_path)
        #for file_name in os.listdir(datesetCreator.rotor_valid_path):
        #    file_path = os.path.join(datesetCreator.rotor_valid_path, file_name)
        #    os.remove(file_path)
        with open(datesetCreator.txt_paths[1], 'w') as txtfile:
            for n, clean_sequence in enumerate(clean_valid_sequences):
                #for snr_valid in datesetCreator.snrs_valid:
                wav_path = valid_dictionary['clean_wav_filenames'][n].split('.')
                #wav_filename = valid_dictionary['speakers'][n] + '_' + str(snr_valid) + '_' + wav_path[0]
                wav_filename = valid_dictionary['speakers'][n] + '_' + wav_path[0]
                txtfile.write(str(os.path.join(datesetCreator.noisy_valid_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(datesetCreator.clean_valid_path, wav_filename + '.wav')) + ' ')
                #txtfile.write(str(os.path.join(datesetCreator.rotor_valid_path, wav_filename + '.csv')) + '\n')
                if np.random.uniform(0, 1) <= datesetCreator.single_rps_percent:
                    single_rotor += 1
                    dregon_id = 0
                else:
                    dregon_id = 1
                if np.random.uniform(0, 1) <= datesetCreator.pure_egonoise_percent:
                    no_bg_noise += 1
                    egonoise_dictionary = datesetCreator.egonoise_rotors(dregon_id=dregon_id)
                    noise_sequence = egonoise_dictionary['egonoise_sequence'][0]
                elif np.random.uniform(0, 1) <= datesetCreator.no_rotor_percent:
                    no_egonoise += 1
                    bg_dictionary = datesetCreator.background_noise(partition='valid')
                    noise_sequence = bg_dictionary['bg_noise_sequence'][0]
                else:
                    egonoise_dictionary = datesetCreator.egonoise_rotors(dregon_id=dregon_id)
                    bg_dictionary = datesetCreator.background_noise(partition='valid')
                    snr_noises = random.choice(datesetCreator.snr_noises)
                    noise_sequence, _, _, _, _ = datesetCreator.mix_snr(egonoise_dictionary['egonoise_sequence'][0], bg_dictionary['bg_noise_sequence'][0], snr_noises)
                if np.random.uniform(0, 1) <= datesetCreator.noise_only_percent:
                    no_speech += 1
                    wavfile.write(os.path.join(datesetCreator.clean_valid_path, wav_filename + '.wav'), datesetCreator.sample_rate, clean_sequence * 0)
                    wavfile.write(os.path.join(datesetCreator.noisy_valid_path, wav_filename + '.wav'), datesetCreator.sample_rate, noise_sequence)
                else:
                    snr_valid = random.choice(datesetCreator.snrs_valid)
                    mixture, signal, _, _, _ = datesetCreator.mix_snr(clean_sequence, noise_sequence, snr_valid)
                    wavfile.write(os.path.join(datesetCreator.clean_valid_path, wav_filename + '.wav'), datesetCreator.sample_rate, signal)
                    wavfile.write(os.path.join(datesetCreator.noisy_valid_path, wav_filename + '.wav'), datesetCreator.sample_rate, mixture)
                
        print('VALID No background noise ', no_bg_noise / (len(clean_valid_sequences) * len(datesetCreator.snrs_valid)) * 100)
        print('VALID No egonoise ', no_egonoise / (len(clean_valid_sequences) * len(datesetCreator.snrs_valid)) * 100)
        print('VALID No speech ', no_speech / (len(clean_valid_sequences) * len(datesetCreator.snrs_valid)) * 100)
        print('VALID Single rotor ', single_rotor /(len(clean_valid_sequences) * len(datesetCreator.snrs_valid)) * 100)

        '''
        no_bg_noise = 0

        test_dictionary = clean_dictionaries[1]
        clean_test_sequences = test_dictionary['clean_sequences']
        for file_name in os.listdir(datesetCreator.clean_test_path):
            file_path = os.path.join(datesetCreator.clean_test_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(datesetCreator.noisy_test_path):
            file_path = os.path.join(datesetCreator.noisy_test_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(datesetCreator.rotor_test_path):
            file_path = os.path.join(datesetCreator.rotor_test_path, file_name)
            os.remove(file_path)
        with open(datesetCreator.txt_paths[2], 'w') as txtfile:
            for n, clean_sequence in enumerate(clean_test_sequences):
                wav_path = test_dictionary['clean_wav_filenames'][n].split('.')
                wav_filename = test_dictionary['speakers'][n] + '_' + wav_path[0]
                txtfile.write(str(os.path.join(datesetCreator.noisy_test_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(datesetCreator.clean_test_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(datesetCreator.rotor_test_path, wav_filename + '.csv')) + '\n')
                if np.random.uniform(0, 1) <= datesetCreator.single_rps_percent:
                    single_rotor += 1
                    dregon_id = 1
                    egonoise_dictionary = datesetCreator.cepo_noise(random.choice([0, 1]), sample_length=len(clean_sequence))
                    egonoise_sequence = egonoise_dictionary['cepo_noise_sequence'][0]
                    rotor_sequence = np.full((datesetCreator.rotors_number, datesetCreator.sample_length), egonoise_dictionary['motor'])
                else:
                    dregon_id = 3
                    egonoise_dictionary = datesetCreator.egonoise_rotors(dregon_id=dregon_id, sample_length=len(clean_sequence))
                    egonoise_sequence = egonoise_dictionary['egonoise_sequence'][0]
                    rotor_sequence = egonoise_dictionary['rotors'][0]
                if np.random.uniform(0, 1) <= datesetCreator.pure_egonoise_percent:
                    noise_sequence = egonoise_sequence
                    no_bg_noise += 1
                else:
                    bg_dictionary = datesetCreator.background_noise(sample_length=len(clean_sequence))
                    snr_noises = random.choice(datesetCreator.snr_noises)
                    noise_sequence, _, _, _, _ = datesetCreator.mix_snr(egonoise_sequence, bg_dictionary['bg_noise_sequence'][0], snr_noises)
                snr_test = random.choice(datesetCreator.snrs_test)
                mixture, signal, _, _, _ = datesetCreator.mix_snr(clean_sequence, noise_sequence, snr_test)
                wavfile.write(os.path.join(datesetCreator.clean_test_path, wav_filename + '.wav'), datesetCreator.sample_rate, signal)
                wavfile.write(os.path.join(datesetCreator.noisy_test_path, wav_filename + '.wav'), datesetCreator.sample_rate, mixture)
                np.savetxt(os.path.join(datesetCreator.rotor_test_path, wav_filename + '.csv'), rotor_sequence, delimiter=',')
        
        print('TEST No background noise ', no_bg_noise / len(clean_test_sequences) * 100)
        print('TEST Single rotor ', single_rotor / len(clean_test_sequences) * 100)
        '''

    else:
        no_bg_noise = 0
        no_egonoise = 0
        no_speech = 0
        single_rotor = 0

        valid_dictionary = clean_dictionaries[0]
        clean_valid_sequences = valid_dictionary['clean_sequences']
        for file_name in os.listdir(datesetCreator.clean_valid_path):
            file_path = os.path.join(datesetCreator.clean_valid_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(datesetCreator.noisy_valid_path):
            file_path = os.path.join(datesetCreator.noisy_valid_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(datesetCreator.rotor_valid_path):
            file_path = os.path.join(datesetCreator.rotor_valid_path, file_name)
            os.remove(file_path)
        with open(datesetCreator.txt_paths[1], 'w') as txtfile:
            for n, clean_sequence in enumerate(clean_valid_sequences):
                #for snr_valid in datesetCreator.snrs_valid:
                wav_path = valid_dictionary['clean_wav_filenames'][n].split('.')
                #wav_filename = valid_dictionary['speakers'][n] + '_' + str(snr_valid) + '_' + wav_path[0]
                wav_filename = valid_dictionary['speakers'][n] + '_' + wav_path[0]
                txtfile.write(str(os.path.join(datesetCreator.noisy_valid_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(datesetCreator.clean_valid_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(datesetCreator.rotor_valid_path, wav_filename + '.csv')) + '\n')
                if np.random.uniform(0, 1) <= datesetCreator.single_rps_percent:
                    single_rotor += 1
                    dregon_id = 0
                else:
                    dregon_id = 1
                if np.random.uniform(0, 1) <= datesetCreator.pure_egonoise_percent:
                    no_bg_noise += 1
                    egonoise_dictionary = datesetCreator.egonoise_rotors(dregon_id=dregon_id)
                    noise_sequence = egonoise_dictionary['egonoise_sequence'][0]
                    rotor_sequence = egonoise_dictionary['rotors'][0]
                elif np.random.uniform(0, 1) <= datesetCreator.no_rotor_percent:
                    no_egonoise += 1
                    bg_dictionary = datesetCreator.background_noise(partition='valid')
                    noise_sequence = bg_dictionary['bg_noise_sequence'][0]
                    rotor_sequence = np.full((datesetCreator.rotors_number, datesetCreator.sample_length), 0)
                else:
                    egonoise_dictionary = datesetCreator.egonoise_rotors(dregon_id=dregon_id)
                    bg_dictionary = datesetCreator.background_noise(partition='valid')
                    snr_noises = random.choice(datesetCreator.snr_noises)
                    noise_sequence, _, _, _, _ = datesetCreator.mix_snr(egonoise_dictionary['egonoise_sequence'][0], bg_dictionary['bg_noise_sequence'][0], snr_noises)
                    rotor_sequence = egonoise_dictionary['rotors'][0]
                if np.random.uniform(0, 1) <= datesetCreator.noise_only_percent:
                    no_speech += 1
                    wavfile.write(os.path.join(datesetCreator.clean_valid_path, wav_filename + '.wav'), datesetCreator.sample_rate, clean_sequence * 0)
                    wavfile.write(os.path.join(datesetCreator.noisy_valid_path, wav_filename + '.wav'), datesetCreator.sample_rate, noise_sequence)
                    np.savetxt(os.path.join(datesetCreator.rotor_valid_path, wav_filename + '.csv'), rotor_sequence, delimiter=',')
                else:
                    snr_valid = random.choice(datesetCreator.snrs_valid)
                    mixture, signal, _, _, _ = datesetCreator.mix_snr(clean_sequence, noise_sequence, snr_valid)
                    wavfile.write(os.path.join(datesetCreator.clean_valid_path, wav_filename + '.wav'), datesetCreator.sample_rate, signal)
                    wavfile.write(os.path.join(datesetCreator.noisy_valid_path, wav_filename + '.wav'), datesetCreator.sample_rate, mixture)
                    np.savetxt(os.path.join(datesetCreator.rotor_valid_path, wav_filename + '.csv'), rotor_sequence, delimiter=',')
            
        print('VALID No background noise ', no_bg_noise / (len(clean_valid_sequences) * len(datesetCreator.snrs_valid)) * 100)
        print('VALID No egonoise ', no_egonoise / (len(clean_valid_sequences) * len(datesetCreator.snrs_valid)) * 100)
        print('VALID No speech ', no_speech / (len(clean_valid_sequences) * len(datesetCreator.snrs_valid)) * 100)
        print('VALID Single rotor ', single_rotor /(len(clean_valid_sequences) * len(datesetCreator.snrs_valid)) * 100)



class DatasetCreator(object):

    def  __init__(self, directory, config):
        '''
        'Abstract' class for creating a trainset and testset.
        Derived classes should implement a specific type of data, i.e. desired signals and noise types.'''

        self.datasets_path = directory
        self.config = config
        self.clean_testset_path = os.path.join(self.datasets_path, self.config['dataset']['clean_testset_wav'])
        self.clean_trainset_path = os.path.join(self.datasets_path, self.config['dataset']['clean_trainset_wav'])
        self.clean_train_path = os.path.join(self.datasets_path, self.config['dataset']['clean_train'])
        self.clean_valid_path = os.path.join(self.datasets_path, self.config['dataset']['clean_valid'])
        self.clean_test_path = os.path.join(self.datasets_path, self.config['dataset']['clean_test'])
        self.noisy_train_path = os.path.join(self.datasets_path, self.config['dataset']['noisy_train'])
        self.noisy_valid_path = os.path.join(self.datasets_path, self.config['dataset']['noisy_valid'])
        self.noisy_test_path = os.path.join(self.datasets_path, self.config['dataset']['noisy_test'])
        self.rotor_train_path = os.path.join(self.datasets_path, self.config['dataset']['rotor_train'])
        self.rotor_valid_path = os.path.join(self.datasets_path, self.config['dataset']['rotor_valid'])
        self.rotor_test_path = os.path.join(self.datasets_path, self.config['dataset']['rotor_test'])
        self.demand_noise_path = os.path.join(self.datasets_path, self.config['dataset']['demand_noise'])
        self.dregon_noise_path = os.path.join(self.datasets_path, self.config['dataset']['dregon_noise'])
        self.cepo_noise_path = os.path.join(self.datasets_path, self.config['dataset']['cepo_noise'])
        self.rotors_number = self.config['dataset']['rotors_number']
        self.sample_rate = self.config['dataset']['sample_rate']
        self.sample_length = self.config['dataset']['sample_length']
        self.snrs_train = self.config['dataset']['snrs_train']
        self.snrs_valid = self.config['dataset']['snrs_valid'] 
        self.snrs_test = self.config['dataset']['snrs_test']
        self.utterances_train = self.config['dataset']['utterances_train']
        self.utterances_valid = self.config['dataset']['utterances_valid']
        self.utterances_test = self.config['dataset']['utterances_test']
        # TODO!!!
        # NUMBER OF THE SPEAKERS FOR THE BINARY ENCODING AND GLOBAL CONDITIONING
        self.speakers_train = self.config['dataset']['speakers_train']
        self.speakers_valid = self.config['dataset']['speakers_valid']
        self.speakers_test = self.config['dataset']['speakers_test']
        # TODO!!!
        self.txt_paths = ["Datasets/train_dataset_rotors.txt", "Datasets/valid_dataset_rotors.txt", "Datasets/test_dataset_rotors.txt"]

        self.merge_or_pick_percent = self.config['dataset']['merge_or_pick_percent']

        self.speech_only_percent = self.config['dataset']['speech_only_percent']
        self.pure_egonoise_percent = self.config['dataset']['pure_egonoise_percent']
        self.noise_only_percent = self.config['dataset']['noise_only_percent']
        self.no_rotor_percent = self.config['dataset']['no_egonoise_percent']
        self.snr_noises = self.config['dataset']['snr_noises']
        self.single_rps_percent = self.config['dataset']['single_rps_percent']


    def on_the_fly(self):

        no_noise = 0
        no_bg_noise = 0
        no_egonoise = 0
        no_speech = 0
        single_rotor = 0

        for file_name in os.listdir(self.clean_train_path):
            file_path = os.path.join(self.clean_train_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(self.noisy_train_path):
            file_path = os.path.join(self.noisy_train_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(self.rotor_train_path):
            file_path = os.path.join(self.rotor_train_path, file_name)
            os.remove(file_path)
        clean_dictionaries = self.clean_speech_speaker_id(on_the_fly="on")
        train_dictionary = clean_dictionaries[0]
        clean_sequences = train_dictionary['clean_sequences']
        with open(self.txt_paths[0], 'w') as txtfile:
            for n, clean_sequence in enumerate(clean_sequences):
                wav_path = train_dictionary['clean_wav_filenames'][n].split('.')
                wav_filename = train_dictionary['speakers'][n] + '_' + wav_path[0]
                txtfile.write(str(os.path.join(self.noisy_train_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(self.clean_train_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(self.rotor_train_path, wav_filename + '.csv')) + '\n')

                if np.random.uniform(0, 1) <= self.single_rps_percent:
                    single_rotor += 1
                    dregon_id = 0
                else:
                    dregon_id = 2
                if np.random.uniform(0, 1) <= self.speech_only_percent:
                    no_noise += 1
                    rotor_sequence = np.full((self.rotors_number, self.sample_length), 0)
                else:
                    if np.random.uniform(0, 1) <= self.pure_egonoise_percent:
                        no_bg_noise += 1
                        egonoise_dictionary = self.egonoise_rotors(dregon_id=dregon_id)
                        noise_sequence = egonoise_dictionary['egonoise_sequence'][0]
                        rotor_sequence = egonoise_dictionary['rotors'][0]
                    elif np.random.uniform(0, 1) <= self.no_rotor_percent:
                        no_egonoise += 1
                        bg_dictionary = self.background_noise(partition='train')
                        noise_sequence = bg_dictionary['bg_noise_sequence'][0]
                        rotor_sequence = np.full((self.rotors_number, self.sample_length), 0)
                    else:
                        egonoise_dictionary = self.egonoise_rotors(dregon_id=dregon_id)
                        bg_dictionary = self.background_noise(partition='train')
                        snr_noises = random.choice(self.snr_noises)
                        noise_sequence, _, _, _, _ = self.mix_snr(egonoise_dictionary['egonoise_sequence'][0], bg_dictionary['bg_noise_sequence'][0], snr_noises)
                        rotor_sequence = egonoise_dictionary['rotors'][0]

                if np.random.uniform(0, 1) <= self.noise_only_percent:
                    no_speech += 1
                    signal = clean_sequence * 0
                else:
                    snr_train = np.random.uniform(self.snrs_train[0], self.snrs_train[1])
                    mixture, signal, _, _, _ = self.mix_snr(clean_sequence, noise_sequence, snr_train)

                wavfile.write(os.path.join(self.clean_train_path, wav_filename + '.wav'), self.sample_rate, signal)
                wavfile.write(os.path.join(self.noisy_train_path, wav_filename + '.wav'), self.sample_rate, mixture)
                np.savetxt(os.path.join(self.rotor_train_path, wav_filename + '.csv'), rotor_sequence, delimiter=',')

            '''
            for n, clean_sequence in enumerate(clean_sequences):
                wav_path = train_dictionary['clean_wav_filenames'][n].split('.')
                wav_filename = train_dictionary['speakers'][n] + '_' + wav_path[0]
                txtfile.write(str(os.path.join(self.noisy_train_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(self.clean_train_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(self.rotor_train_path, wav_filename + '.csv')) + '\n')
                if np.random.uniform(0, 1) <= self.single_rps_percent:
                    single_rotor += 1
                    dregon_id = 0
                else:
                    dregon_id = 2
                if np.random.uniform(0, 1) <= self.speech_only_percent:
                    no_noise += 1
                    wavfile.write(os.path.join(self.clean_train_path, wav_filename + '.wav'), self.sample_rate, clean_sequence)
                    wavfile.write(os.path.join(self.noisy_train_path, wav_filename + '.wav'), self.sample_rate, clean_sequence)
                    rotor_sequence = np.full((self.rotors_number, self.sample_length), 0)
                    np.savetxt(os.path.join(self.rotor_train_path, wav_filename + '.csv'), rotor_sequence, delimiter=',')
                else:
                    if np.random.uniform(0, 1) <= self.pure_egonoise_percent:
                        no_bg_noise += 1
                        egonoise_dictionary = self.egonoise_rotors(dregon_id=dregon_id)
                        noise_sequence = egonoise_dictionary['egonoise_sequence'][0]
                        rotor_sequence = egonoise_dictionary['rotors'][0]

                    elif np.random.uniform(0, 1) <= self.no_rotor_percent:
                        no_egonoise += 1
                        bg_dictionary = self.background_noise()
                        noise_sequence = bg_dictionary['bg_noise_sequence'][0]
                        rotor_sequence = np.full((self.rotors_number, self.sample_length), 0)
                    else:
                        egonoise_dictionary = self.egonoise_rotors(dregon_id=dregon_id)
                        bg_dictionary = self.background_noise()
                        snr_noises = random.choice(self.snr_noises)
                        noise_sequence, _, _, _, _ = self.mix_snr(egonoise_dictionary['egonoise_sequence'][0], bg_dictionary['bg_noise_sequence'][0], snr_noises)
                        rotor_sequence = egonoise_dictionary['rotors'][0]
                    if np.random.uniform(0, 1) <= self.noise_only_percent:
                        no_speech += 1
                        wavfile.write(os.path.join(self.clean_train_path, wav_filename + '.wav'), self.sample_rate, clean_sequence * 0)
                        wavfile.write(os.path.join(self.noisy_train_path, wav_filename + '.wav'), self.sample_rate, noise_sequence)
                        np.savetxt(os.path.join(self.rotor_train_path, wav_filename + '.csv'), rotor_sequence, delimiter=',')
                    else:
                        snr_train = np.random.uniform(self.snrs_train[0], self.snrs_train[1])
                        mixture, signal, _, _, _ = self.mix_snr(clean_sequence, noise_sequence, snr_train)
                        wavfile.write(os.path.join(self.clean_train_path, wav_filename + '.wav'), self.sample_rate, signal)
                        wavfile.write(os.path.join(self.noisy_train_path, wav_filename + '.wav'), self.sample_rate, mixture)
                        np.savetxt(os.path.join(self.rotor_train_path, wav_filename + '.csv'), rotor_sequence, delimiter=',')
            '''
        print('TRAIN No noise ', no_noise / len(clean_sequences) * 100)
        print('TRAIN No background noise ', no_bg_noise / len(clean_sequences) * 100)
        print('TRAIN No egonoise ', no_egonoise / len(clean_sequences) * 100)
        print('TRAIN No speech ', no_speech / len(clean_sequences) * 100)
        print('TRAIN Single rotor ', single_rotor / len(clean_sequences) * 100)


    def test_at_SNR(self, snr_test):

        no_bg_noise = 0
        single_rotor = 0
        test_dictionary = self.clean_speech_speaker_id(on_the_fly="fixed")
        test_dictionary = test_dictionary[0]
        clean_test_sequences = test_dictionary['clean_sequences']
        for file_name in os.listdir(self.clean_test_path):
            file_path = os.path.join(self.clean_test_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(self.noisy_test_path):
            file_path = os.path.join(self.noisy_test_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(self.rotor_test_path):
            file_path = os.path.join(self.rotor_test_path, file_name)
            os.remove(file_path)
        with open(self.txt_paths[2], 'w') as txtfile:
            for n, clean_sequence in enumerate(clean_test_sequences):
                wav_path = test_dictionary['clean_wav_filenames'][n].split('.')
                wav_filename = test_dictionary['speakers'][n] + '_' + wav_path[0]
                txtfile.write(str(os.path.join(self.noisy_test_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(self.clean_test_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(self.rotor_test_path, wav_filename + '.csv')) + '\n')
                if np.random.uniform(0, 1) <= self.single_rps_percent:
                    single_rotor += 1
                    dregon_id = 1
                    egonoise_dictionary = self.cepo_noise(random.choice([0, 1]), sample_length=len(clean_sequence))
                    egonoise_sequence = egonoise_dictionary['cepo_noise_sequence'][0]
                    rotor_sequence = np.full((self.rotors_number, self.sample_length), egonoise_dictionary['motor'])
                else:
                    dregon_id = 3
                    egonoise_dictionary = self.egonoise_rotors(dregon_id=dregon_id, sample_length=len(clean_sequence))
                    egonoise_sequence = egonoise_dictionary['egonoise_sequence'][0]
                    rotor_sequence = egonoise_dictionary['rotors'][0]
                if np.random.uniform(0, 1) <= self.pure_egonoise_percent:
                    noise_sequence = egonoise_sequence
                    no_bg_noise += 1
                else:
                    bg_dictionary = self.background_noise(partition='test', sample_length=len(clean_sequence))
                    snr_noises = random.choice(self.snr_noises)
                    noise_sequence, _, _, _, _ = self.mix_snr(egonoise_sequence, bg_dictionary['bg_noise_sequence'][0], snr_noises)
                #snr_test = random.choice(self.snrs_test)
                mixture, signal, _, _, _ = self.mix_snr(clean_sequence, noise_sequence, snr_test)
                wavfile.write(os.path.join(self.clean_test_path, wav_filename + '.wav'), self.sample_rate, signal)
                wavfile.write(os.path.join(self.noisy_test_path, wav_filename + '.wav'), self.sample_rate, mixture)
                np.savetxt(os.path.join(self.rotor_test_path, wav_filename + '.csv'), rotor_sequence, delimiter=',')
        
        print('TEST No background noise ', no_bg_noise / len(clean_test_sequences) * 100)
        print('TEST Single rotor ', single_rotor / len(clean_test_sequences) * 100)



    def test_at_SNR_no_rotors(self, snr_test):

        no_bg_noise = 0

        test_dictionary = self.clean_speech_speaker_id(on_the_fly="fixed")
        test_dictionary = test_dictionary[0]
        clean_test_sequences = test_dictionary['clean_sequences']
        for file_name in os.listdir(self.clean_test_path):
            file_path = os.path.join(self.clean_test_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(self.noisy_test_path):
            file_path = os.path.join(self.noisy_test_path, file_name)
            os.remove(file_path)
        for file_name in os.listdir(self.rotor_test_path):
            file_path = os.path.join(self.rotor_test_path, file_name)
            os.remove(file_path)
        with open(self.txt_paths[2], 'w') as txtfile:
            for n, clean_sequence in enumerate(clean_test_sequences):
                wav_path = test_dictionary['clean_wav_filenames'][n].split('.')
                wav_filename = test_dictionary['speakers'][n] + '_' + wav_path[0]
                txtfile.write(str(os.path.join(self.noisy_test_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(self.clean_test_path, wav_filename + '.wav')) + ' ')
                txtfile.write(str(os.path.join(self.rotor_test_path, wav_filename + '.csv')) + '\n')
                if np.random.uniform(0, 1) <= self.single_rps_percent:
                    single_rotor += 1
                    dregon_id = 1
                    egonoise_dictionary = self.cepo_noise(random.choice([0, 1]), sample_length=len(clean_sequence))
                    egonoise_sequence = egonoise_dictionary['cepo_noise_sequence'][0]
                    rotor_sequence = np.full((self.rotors_number, self.sample_length), egonoise_dictionary['motor'])
                else:
                    dregon_id = 3
                    egonoise_dictionary = self.egonoise_rotors(dregon_id=dregon_id, sample_length=len(clean_sequence))
                    egonoise_sequence = egonoise_dictionary['egonoise_sequence'][0]
                    rotor_sequence = egonoise_dictionary['rotors'][0]
                if np.random.uniform(0, 1) <= self.pure_egonoise_percent:
                    noise_sequence = egonoise_sequence
                    no_bg_noise += 1
                else:
                    bg_dictionary = self.background_noise(partition='test', sample_length=len(clean_sequence))
                    snr_noises = random.choice(self.snr_noises)
                    noise_sequence, _, _, _, _ = self.mix_snr(egonoise_sequence, bg_dictionary['bg_noise_sequence'][0], snr_noises)
                #snr_test = random.choice(self.snrs_test)
                mixture, signal, _, _, _ = self.mix_snr(clean_sequence, noise_sequence, snr_test)
                wavfile.write(os.path.join(self.clean_test_path, wav_filename + '.wav'), self.sample_rate, signal)
                wavfile.write(os.path.join(self.noisy_test_path, wav_filename + '.wav'), self.sample_rate, mixture)
                #np.savetxt(os.path.join(self.rotor_test_path, wav_filename + '.csv'), rotor_sequence, delimiter=',')
        
        print('TEST No background noise ', no_bg_noise / len(clean_test_sequences) * 100)
        print('TEST Single rotor ', single_rotor / len(clean_test_sequences) * 100)


    def clean_speech_speaker_id(self, on_the_fly):
        '''
        Retrieve clean wav path, wav filename, speaker id and clean sequence downsampled and mono.
        Output list of three dictionaries: train, valid and test.'''
        
        if  on_the_fly == "on":
            list_num = 1
        else:
            list_num = 2
        clean_data_dics = [{'clean_wavpaths': [], 'clean_wav_filenames': [], 'speakers': [], 'clean_sequences': []} for _ in range(list_num)]
        clean_paths = [self.clean_trainset_path, self.clean_testset_path]
        input_index = 0
        output_index = 0
        breakTrue = False
        outerBreak = False
        for subdir in os.listdir(clean_paths[input_index]):
            subdir1 = os.path.join(clean_paths[input_index], subdir)
            for subdir2 in os.listdir(subdir1):
                speaker_dir = os.path.join(subdir1, subdir2)
                files_speaker = os.listdir(speaker_dir)
                for data in files_speaker:
                    if data[-3:] == 'wav':
                        wavpath = os.path.join(speaker_dir, data)
                        current_dict = clean_data_dics[output_index]
                        current_dict['clean_wavpaths'].append(wavpath)
                        current_dict['clean_wav_filenames'].append(data)
                        current_dict['speakers'].append(subdir2)
                        downsampled = self.downsample(wavpath)
                        if output_index == 2:
                            current_dict['clean_sequences'].append(downsampled)
                        else:
                            half_padding = int((self.sample_length - len(downsampled)) * 0.5)
                            if half_padding >= 0:
                                #print(data)
                                #print(half_padding)
                                downsampled = np.pad(downsampled, (half_padding + 1, half_padding + 1), 'constant', constant_values=(0, 0))
                            cut, _ = self.wav_cut(downsampled)
                            current_dict['clean_sequences'].append(cut)
                    if list_num == 1:
                        if len(clean_data_dics[0]['clean_sequences']) >= self.utterances_train:
                            breakTrue = True
                            outerBreak = True
                            break
                    else:
                        '''
                        if len(clean_data_dics[0]['clean_sequences']) >= self.utterances_train:
                            output_index = 1
                        if len(clean_data_dics[1]['clean_sequences']) >= self.utterances_valid:
                            output_index = 2
                            input_index = 1
                            breakTrue = True
                        if len(clean_data_dics[2]['clean_sequences']) >= self.utterances_test:
                            breakTrue = True
                            outerBreak = True
                            break
                        '''
                        if len(clean_data_dics[0]['clean_sequences']) >= self.utterances_valid:
                            output_index = 1
                        if len(clean_data_dics[1]['clean_sequences']) >= self.utterances_test:
                            output_index = 2
                            input_index = 1
                            breakTrue = True
                            outerBreak = True
                            break
                if breakTrue:
                    break
            if outerBreak:
                    break
        return clean_data_dics


    def downsample(self, wav_path):
        '''
        Load wav from wav_path and return downsampled sequence.'''

        if wav_path.endswith('.wav'):
            sampling_rate, data = wavfile.read(wav_path)
            if len(data.shape) > 1:
                sequence = self.get_mono_wav(data)
            else:
                sequence = data
        elif wav_path.endswith('.flac'):
            sequence, sampling_rate = sf.read(wav_path)
        samples = round(len(sequence) * float(self.sample_rate) / sampling_rate)
        sequence = sps.resample(sequence, samples)
        return sequence
    
    

    def get_mono_wav(self, data):
        '''
        Return normalized mono sequence from multichannel sequence, 
        merging all channels or picking randomly one channel based on the given percentage.'''

        if self.merge_or_pick_percent > 0:
                if np.random.uniform(0, 1) <= self.merge_or_pick_percent:
                    merged_sequence = np.zeros([data.shape[0]])
                    for i in range(0, data.shape[1]):
                            merged_sequence += normalize(data[:,i]) / data.shape[1]
                            sequence = merged_sequence
                else:
                    channel = np.random.randint(0, data.shape[1])
                    rand_mono_sequence = data[:, channel]
                    rand_mono_sequence = normalize(rand_mono_sequence)
                    sequence = rand_mono_sequence
        return sequence
    

    def wav_cut(self, sequence, sample_length=None):
        '''
        Cut sequence with a mean length of 3 seconds and standard deviation of 1 second,
        at a random point.'''
        if sample_length == None:
            sample_length = self.sample_length
        assert len(sequence) >= sample_length, "Sequence must be longer than the given sample length."

        new_length_sequence = sample_length
        length_sequence = sequence.shape[0]
        surplus = length_sequence - new_length_sequence
        offset = 0
        if surplus > 0:
            offset = np.random.randint(0, surplus)
        sequence = np.asarray(sequence[offset:offset + new_length_sequence], dtype=np.float32)
        return sequence, offset
    

    def wav_cut_UAV(self, sequence, sample_length=None):
        '''
        Cut sequence with a mean length of 3 seconds and standard deviation of 1 second,
        at a random point.'''
        if sample_length == None:
            sample_length = self.sample_length
        assert len(sequence) >= sample_length + 30 * self.sample_rate, "DREGON Sequence must be longer than the given sample length + 30 seconds."
        original_length = len(sequence)
        new_length_sequence = sample_length
        sequence = sequence[10 * self.sample_rate: -20 * self.sample_rate]
        length_sequence = sequence.shape[0]
        surplus = length_sequence - new_length_sequence
        offset = 0
        if surplus > 0:
            offset = np.random.randint(0, surplus)
        sequence = np.asarray(sequence[offset:offset + new_length_sequence], dtype=np.float32)
        return sequence, offset, original_length


    def mix_snr(self, signal_sequence, noise_sequence, snr):
        '''
        Mix signal and noise at the given snr and returns mix, signal and noise with the effective gain.'''

        assert len(signal_sequence) == len(noise_sequence), "Noise and signal sequences must have the same length."

        noise_sequence = noise_sequence.astype(np.float32)
        signal_sequence = signal_sequence.astype(np.float32)

        signal_energy = np.mean(signal_sequence ** 2)
        noise_energy = np.mean(noise_sequence ** 2)
        try:
            gain = np.sqrt(10.0 ** (-snr / 10) * signal_energy / noise_energy)
        except ValueError:
            return signal_sequence

        signal_gain = np.sqrt(1 / (1 + gain ** 2))
        noise_gain = np.sqrt(gain ** 2 / (1 + gain ** 2))
        #signal_gain = 1
        #noise_gain = gain
        signal = signal_gain * signal_sequence
        noise = noise_gain * noise_sequence
        mixture = signal + noise
        mixture = np.asarray(mixture, dtype=np.float32)
        signal = np.asarray(signal, dtype=np.float32)
        noise = np.asarray(noise, dtype=np.float32)
        #print('nominal snr', snr)
        #print('real snr', snr_db(mixture - noise, noise))
        return mixture, signal, noise, signal_gain, noise_gain


    def egonoise_rotors(self, dregon_id, sample_length=None):
        '''
        Extract egonoise and motor measurement sequences from .mat files.
        dregon_id: identification of room 2 (0: for training), room 1 (1: for validation), individual motors (2), files for testing (3) '''

        if sample_length == None:
            sample_length = self.sample_length
        egonoise_data_dic = {'egonoise_wavpath' : [], 'egonoise_wav_filename' : [], 'rotors' : [], 'egonoise_sequence' : []}
        egonoise_dirs = []
        for egonoise_dir in os.listdir(self.dregon_noise_path):
            if dregon_id == 0: #
                if egonoise_dir[-1] == '2':
                    egonoise_dirs.append(egonoise_dir)
            elif dregon_id == 1:
                if egonoise_dir[-1] == '1':
                    egonoise_dirs.append(egonoise_dir)
            elif dregon_id == 3:
                if egonoise_dir[-1] == '3':
                    egonoise_dirs.append(egonoise_dir)
            else:
                if egonoise_dir[-1] == 's':
                    subdir = os.path.join(self.dregon_noise_path, egonoise_dir)
                    listdir = os.listdir(subdir)
                    subsubdir = listdir[0]
                    egonoise_dirs.append(os.path.join(subdir, subsubdir))
        subdir = random.choice(egonoise_dirs)
        #for subdir in os.listdir(self.dregon_noise_path):
        subdir_path = os.path.join(self.dregon_noise_path, subdir)
        if dregon_id == 2:
            data = random.choice(os.listdir(subdir_path))
            data_path = os.path.join(subdir_path, data)
            egonoise_data_dic['egonoise_wavpath'].append(data_path)
            egonoise_data_dic['egonoise_wav_filename'].append(data)
            downsampled = self.downsample(data_path)
            downsampled = downsampled[5 * self.sample_rate : - 10 * self.sample_rate]
            half_padding = int((sample_length - len(downsampled)) * 0.5)
            if half_padding >= 0:
                #print(data)
                #print(half_padding)
                downsampled = np.pad(downsampled, (half_padding + 1, half_padding + 1), 'constant', constant_values=(0, 0))
            cut, _ = self.wav_cut(downsampled, sample_length)
            egonoise_data_dic['egonoise_sequence'].append(cut)
            rotor_choice = data[5]
            rps = data[-6:-4]
            motors_upsampled = np.zeros((self.rotors_number, sample_length))
            try:
                rotor = int(rotor_choice) - 1
                motors_upsampled[rotor] = [int(rps)] * sample_length
            except ValueError:
                motors_upsampled = np.full((self.rotors_number, sample_length), int(rps))
            egonoise_data_dic['rotors'].append(motors_upsampled)
        else:
            for data in os.listdir(subdir_path):
                if data[-3:] == 'wav':
                    data_path = os.path.join(subdir_path, data)
                    egonoise_data_dic['egonoise_wavpath'].append(data_path)
                    egonoise_data_dic['egonoise_wav_filename'].append(data)
                    downsampled = self.downsample(data_path)
                    half_padding = int((sample_length + 30 * self.sample_rate - len(downsampled)) * 0.5)
                    if half_padding >= 0:
                        #print(data_path)
                        #print(half_padding)
                        downsampled = np.pad(downsampled, (half_padding + 1, half_padding + 1), 'constant', constant_values=(0, 0))
                    cut, offset, original_length = self.wav_cut_UAV(downsampled, sample_length)
                    egonoise_data_dic['egonoise_sequence'].append(cut)
            for data in os.listdir(subdir_path):
                if data[-10:] == 'motors.mat':
                    motors = loadmat(os.path.join(subdir_path, data))['motor'][0]
                    motors = motors[0]
                    motors = motors[-1]
                    motors_upsampled = np.zeros((self.rotors_number, sample_length))
                    for rotor in range(self.rotors_number):
                        rotor_upsampled = sps.resample(motors[rotor, :], original_length)
                        rotor_upsampled = rotor_upsampled[10 * self.sample_rate: -20 * self.sample_rate]
                        motors_upsampled[rotor, :] = rotor_upsampled[offset: offset + sample_length]
                    egonoise_data_dic['rotors'].append(motors_upsampled)
        return egonoise_data_dic

    
    def background_noise(self, partition, sample_length=None):
        '''
        Extract background noise from multichannel DEMAND dataset. '''

        if sample_length == None:
            sample_length = self.sample_length
        bg_noise_data_dic= {'bg_noise_wavpath' : [], 'bg_noise_wav_filename' : [], 'ambient' : [], 'bg_noise_sequence' : []}
        subdir = random.choice(os.listdir(self.demand_noise_path))
        subdir_path = os.path.join(self.demand_noise_path, subdir)
        listdir = os.listdir(os.path.join(self.demand_noise_path, subdir))
        subdir = listdir[0]
        subdir_path = os.path.join(subdir_path, subdir)
        list_demand = os.listdir(subdir_path)
        if partition == 'train':
            data = random.choice(list_demand[:3])
        elif partition == 'valid':
            data = random.choice(list_demand[3:6])
        else:
            data = random.choice(list_demand[6:])
        path_data = os.path.join(subdir_path, data)
        bg_noise_data_dic['bg_noise_wavpath'].append(path_data)
        bg_noise_data_dic['bg_noise_wav_filename'].append(data)
        bg_noise_data_dic['ambient'].append(subdir)
        downsampled = self.downsample(path_data)
        cut, _ = self.wav_cut(downsampled, sample_length)
        bg_noise_data_dic['bg_noise_sequence'].append(cut)
        return bg_noise_data_dic
    

    def cepo_noise(self, all_or_single, sample_length=None):
        '''
        Extract background noise from multichannel DEMAND dataset. 
        all_or_single: identification of one rotor (0), or all (1)'''

        if sample_length == None:
            sample_length = self.sample_length
        cepo_noise_data_dic= {'cepo_noise_wavpath' : [], 'cepo_noise_wav_filename' : [], 'motor' : [], 'cepo_noise_sequence' : []}
        listdir = os.listdir(self.cepo_noise_path)
        if all_or_single == 0:
            rotor_choice = str(random.choice(range(1, self.rotors_number + 1)))
            single_rotor_data = []
            for data in listdir:
                if data[6] == rotor_choice:
                    single_rotor_data.append(data)
            cepo_data = random.choice(single_rotor_data)
        else:
            all_rotors_data = []
            rotor_choice = 'A'
            for data in listdir:
                if data[6] == rotor_choice:
                    all_rotors_data.append(data)
            cepo_data = random.choice(all_rotors_data)
        path_data = os.path.join(self.cepo_noise_path, cepo_data)
        cepo_noise_data_dic['cepo_noise_wavpath'].append(path_data)
        cepo_noise_data_dic['cepo_noise_wav_filename'].append(cepo_data)
        cepo_noise_data_dic['motor'].append(0) # TODO
        sequence, sampling_rate = sf.read(path_data)
        if len(sequence.shape) > 1:
            sequence = self.get_mono_wav(data)
        samples = round(len(sequence) * float(self.sample_rate) / sampling_rate)
        downsampled = sps.resample(sequence, samples)
        downsampled = downsampled[25 * self.sample_rate : - 25 * self.sample_rate]
        cut, _ = self.wav_cut(downsampled, sample_length)
        cepo_noise_data_dic['cepo_noise_sequence'].append(cut)
        return cepo_noise_data_dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Dataset with Clean speech, Egonoise and Background Noise")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).")
    parser.add_argument("-G", "--generate", required=True, type=str, help="Generate on the fly (on) or fixed (fix), or for SEGAN (SEGAN)")
    args = parser.parse_args()

    configuration = json.load(open(args.configuration))
    configuration["config_path"] = args.configuration

    main(configuration, on_the_fly=args.generate)


def extract_rotors_beamformed(dregon_filename, data, offset, Lh_order, P_order, rotors_number=4):
        '''
        Extract motor measurement sequences from .mat files, aligned to the beamformed output
        dregon_filename: filename of the file of the DREGON dataset
        data: original multichannel noisy or noise sequence
        noisy_sequence: sequence extracted from the WAV file
        P_order: order of the Whittaker-Shannon (sinc function) interpolation formula
        Lh_order: order of the filter (number of samples in the past considered)
        rotors_number: number of rotors of the UAV
         '''

        L = 2 * P_order + Lh_order
        original_length = len(data[:, 0])
        if L == 0:
            length_out = original_length
        else:
            length_out = original_length - L + 1
        if 'Motor' in dregon_filename:
            rps = dregon_filename[-6:-4]
            rotor_choice = dregon_filename[-8]
            motors_upsampled = np.zeros((rotors_number, length_out))
            try:
                rotor = int(rotor_choice) - 1
                motors_upsampled[rotor] = [int(rps)] * length_out
            except ValueError:
                motors_upsampled = np.full((rotors_number, length_out), int(rps))
        else:
            rotors_filename = dregon_filename[:-4] + '_motors.mat'
            motors = loadmat(rotors_filename)['motor'][0]
            motors = motors[0]
            motors = motors[-1]
            motors_upsampled = np.zeros((rotors_number, length_out))
            for rotor in range(rotors_number):
                rotor_upsampled = sps.resample(motors[rotor, :], original_length)
                motors_upsampled[rotor, :] = rotor_upsampled[P_order:original_length - L + P_order + 1]
        return motors_upsampled