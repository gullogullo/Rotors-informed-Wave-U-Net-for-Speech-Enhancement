# Rotor-informed Wave-U-Net for Speech Enhancement

PyTorch implementation of [Wave-U-Net](https://arxiv.org/abs/1806.03185) for speech enhancement with additional CSV file with rotor rps time series.

## Dependencies

librosa, numpy, torch, scipy, tqdm, soundpy, json5, pesq, pystoi, torchmetrics, soundfile

# Clone
git clone https://github.com/gullogullo/Rotors-informed-Wave-U-Net-for-Speech-Enhancement.git
```

## Usage 

There are the SEGAN model, the Wave-U-Net model, and the rotor-informed Wave-U-Net model

- Entry file for training Wave-U-Net models: `train.py`
- Entry file for enhance noisy speech with Wave-U-Net models: `enhancement.py` and `enhancement_rotors.py`
- Entry file for training SEGAN model: `segan_main.py`
- Entry file for enhance noisy speech with SEGAN model: `segan_enhancement.py`