import os
from torch.utils.data import Dataset
import numpy as np
import librosa
import scipy.signal as sg


class WaveformDataset(Dataset):
    def __init__(self, dataset, limit=None, offset=0, sample_length=16384):
        """Construct dataset for enhancement.
        Args:
            dataset (str): *.txt. The path of the dataset list file. See "Notes."
            limit (int): Return at most limit files in the list. If None, all files are returned.
            offset (int): Return files starting at an offset within the list. Use negative values to offset from the end of the list.

        Notes:
            dataset list file：
            <noisy_1_path>
            <noisy_2_path>
            ...
            <noisy_n_path>

            e.g.
            /enhancement/noisy/a.wav
            /enhancement/noisy/b.wav
            ...

        Return:
            (mixture signals, filename)
        """
        super(WaveformDataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]
        #print('dataset_list', dataset_list)

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_path = self.dataset_list[item]
        #mixture_path = mixture_path.split('waveUNet/')
        #mixture_path = mixture_path[1]
        #mix_path = mixture_path[1].split(' ')
        #mix_path = mix_path[0]
        #clean_path = mixture_path[2].split(' ')
        #clean_path = clean_path[0]
        #rotor_path = mixture_path[3]
        
        #name = os.path.splitext(os.path.basename(mixture_path[1]))[0]
        name = os.path.splitext(os.path.basename(mixture_path))[0]
        mixture_list = mixture_path.split(' ')
        mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_list[0])), sr=None)
        clean, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_list[1])), sr=None)
        rotor = np.loadtxt(os.path.abspath(os.path.expanduser(mixture_list[-1])), dtype=str)
        #mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mix_path)), sr=None)
        #clean, _ = librosa.load(os.path.abspath(os.path.expanduser(clean_path)), sr=None)
        #rotor = np.loadtxt(os.path.abspath(os.path.expanduser(rotor_path)), dtype=str)
        #print('rotor 1', rotor.shape)
        rotor_array = []
        for element in rotor:
            rotor_strings = element.split(',')
            rotor_floats = [float(string) for string in rotor_strings]
            rotor_array.append(rotor_floats)
        rotor = np.array(rotor_array)
        #print('rotor 2', rotor.shape)
        '''
        if rotor[0, 0]:
            rotors_number = rotor.shape[1]
            rotor_length = mixture.shape[-1]
            rotor_data = np.random.rand(rotors_number, rotor_length)
            for j in range(rotors_number):
                rotor_data[j] = sg.resample(rotor[:, j], rotor_length)
            rotor = rotor_data
        else:
            rotor = None
        print('rotor 3', rotor.shape)
        '''
        #return mixture.reshape(1, -1), rotor, name
        return mixture.reshape(1, -1), clean.reshape(1, -1), rotor, name