# Rotor-informed Wave-U-Net for Speech Enhancement

PyTorch implementation of [Wave-U-Net](https://arxiv.org/abs/1806.03185) for speech enhancement with additional CSV file with rotor rps time series.

## Dependencies

librosa, numpy, torch, scipy, tqdm, soundpy, json5, pesq, pystoi, torchmetrics, soundfile

## Clone
git clone https://github.com/gullogullo/Rotors-informed-Wave-U-Net-for-Speech-Enhancement.git

## Usage 

There are the SEGAN model, the Wave-U-Net model, and the rotor-informed Wave-U-Net model

- Entry file for training Wave-U-Net models: `train.py`
- Entry file for enhance noisy speech with Wave-U-Net models: `enhancement.py` and `enhancement_rotors.py`
- Entry file for training SEGAN model: `segan_main.py`
- Entry file for enhance noisy speech with SEGAN model: `segan_enhancement.py`

# Examples

[MMGG0_SI2339_Noisy.mov.webm](https://github.com/gullogullo/Rotors-informed-Wave-U-Net-for-Speech-Enhancement/assets/40691310/32b841d6-d924-40de-a76e-dbda6519f459)
[MMGG0_SI2339_Enhanced.mov.webm](https://github.com/gullogullo/Rotors-informed-Wave-U-Net-for-Speech-Enhancement/assets/40691310/cd058128-32fb-407b-91af-f7efe805f7b6)

[MTRR0_SA2_Noisy.mov.webm](https://github.com/gullogullo/Rotors-informed-Wave-U-Net-for-Speech-Enhancement/assets/40691310/05960495-7b4b-42c1-8f81-e107a226cd07)
[MTRR0_SA2_Enhanced.mov.webm](https://github.com/gullogullo/Rotors-informed-Wave-U-Net-for-Speech-Enhancement/assets/40691310/0a75eec6-8db2-42e7-a485-9d79d0507340)
