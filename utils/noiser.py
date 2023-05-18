from utils.general_function import get_file_list_from_dir
from utils.audio_dataloader import audio_dataLoader
import os
import math
import numpy as np
import torch
import torchaudio

def add_noise(path, path_final_folder, no_STD, sr = 16000):
    
    files = get_file_list_from_dir(path, filetype= '')

    for w in files:

        if not os.path.isdir(path_final_folder + w):
            os.mkdir(path_final_folder + w)

    for x in files:

        temp = path + x +'/'
        next_files = get_file_list_from_dir(temp)

        for y in next_files:

            final_path = temp + y

            DL = audio_dataLoader(final_path)
            DL.resample(sr)
            signal = DL.ynumpy

            STD_n = (math.sqrt(np.mean(signal**2))) * no_STD

            noise=np.random.normal(0, STD_n, signal.shape[0])

            noised_signal = signal + noise

            noised_signal_edited = np.expand_dims(noised_signal, axis = 0)
            noised_signal_edited.shape

            noised_tensor = torch.from_numpy(noised_signal_edited)

            torchaudio.save(path_final_folder+x+'/'+y, noised_tensor.type(torch.float32), sr)

def add_chatter(path, path_final_folder, path_chatter_audio, sr = 16000, chatter_volumn = 0.15):

    resample_noise_chatter(path_chatter_audio, sr)

    files = get_file_list_from_dir(path, '')

    for w in files:

        if not os.path.isdir(path_final_folder + w):
            os.mkdir(path_final_folder + w)

    for x in files:

        temp = path + x +'/'
        next_files = get_file_list_from_dir(temp)

        for y in next_files:

            final_path = temp + y

            DL_audio = audio_dataLoader(final_path)
            DL_audio.resample(16000)
            DL_noise = audio_dataLoader(path_chatter_audio)
            noise_numpy = DL_noise.ynumpy * chatter_volumn
            audio_numpy = DL_audio.ynumpy

            target_shape = audio_numpy.shape[0]

            elongated_noise = np.repeat([noise_numpy], target_shape // noise_numpy.shape[0] + 1, axis=0)
            elongated_noise = np.concatenate(elongated_noise, axis=0)[:target_shape]

            final_audio = elongated_noise + audio_numpy
            final_audio = np.expand_dims(final_audio, axis = 0)
            final_tensor = torch.from_numpy(final_audio)

            torchaudio.save(path_final_folder+x+'/'+y, final_tensor.type(torch.float32), 16000)

    return
    
def resample_noise_chatter(path_to_noise_chatter, sr):

    waveform, sample_rate = torchaudio.load(path_to_noise_chatter)
    resampled = torchaudio.transforms.Resample(sample_rate, sr)(waveform)

    torchaudio.save(path_to_noise_chatter, resampled.type(torch.float32), sr)

    return