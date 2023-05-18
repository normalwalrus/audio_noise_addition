import torch
import torchaudio
from torchaudio import transforms
import numpy as np
import librosa
import librosa.display
import random
import matplotlib.pyplot as plt
from logzero import logger

class audio_dataLoader:

    """
    Class is used to load in wav files to allow for data augmentation and MFCC production
    """

    def __init__(self, audio):
        """
        Initialise the dataloader

        Parameters
        ----------
            audio : String
                Path to wav file
        """

        self.y = torchaudio.load(audio, normalize = True)
        self.sr = self.y[1]
        print(self.y)
        self.ynumpy = (self.y[0][0]).numpy()

    def rechannel(self, new_channel):
        """
        Rechannel the wav signal to mono (0) or stereo (1) 

        Parameters
        ----------
            new_channel : Integer
                0 = Mono, 1 = Stereo

        Returns
        ----------
            self.y : Numpy array
                New features in numpy form with the new channel
        """
        sig, sr = self.y
        if (sig.shape[0] == new_channel):
            return self.y
        if (new_channel == 1):
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])

        self.y = (resig, sr)
        self.ynumpy = self.y[0][0].numpy()
        return self.y

    def resample(self, newsr):
        """
        Resample the wav signal 

        Parameters
        ----------
            newsr : Integer
                The new sampling rate that is needed

        """
        sig, sr = self.y
        if (sr == newsr):
            return self.y
        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        self.y = (resig, newsr)
        self.ynumpy = self.y[0][0].numpy()
        self.sr = newsr
        return
    
    def pad_trunc(self, max_ms):
        """
        Pads the audio with no sound to make up the desired length of audio

        Parameters
        ----------
            max_ms : Integer
                Length of audio that is desired in miliseconds
        Returns
        ----------
            self.y : Numpy array
                New features in numpy form with the length

        """
        sig, sr = self.y
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms
        if (sig_len > max_len):
            sig = sig[:,:max_len]
        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)

        self.y = (sig, sr)
        self.ynumpy = self.y[0][0].numpy()
        return self.y
    
    def time_shift(self, shift_limit):
        """
        Shifts the audio by a random amount of time

        Parameters
        ----------
            shift_limit : Integer
                Limit to how much shifting is done
        Returns
        ----------
            self.y : Numpy array
                New features in numpy form with the shift completed

        """
        sig,sr = self.y
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)

        self.y = (sig.roll(shift_amt), sr)
        self.ynumpy = self.y[0][0].numpy()
        return self.y

    def spectro_gram(self, n_mels=64, n_fft=1024, hop_len=None, condensed = False):
        """
        Produces a 2D array of the spectrogram from the wav file in the dataloader

        Parameters
        ----------
            n_mels : Integer
                Number of mels used in the spectrogram production
            n_fft : Integer
                Number of fast fourier transforms used in the spectrogram production
            hop_len : Integer
                number of audio samples between adjacent FFT columns.
            condensed : Boolean
                Whether the resultant Spectrogram should be averaged out or not
        Returns
        ----------
            spec : Numpy array
                2D Numpy array of the resultant Spectrogram

        """
        sig,sr = self.y
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

        if condensed:
            spec = np.array(spec[0]).mean(axis=0)

        return np.array(spec)
    
    def MFCC_extraction(self , n_mfcc = 40, mean = True, remix = True, align_zero = True, max_ms = 5000):
        """
        Produces MFCC from the wav file in the dataloader

        Parameters
        ----------
            n_mfcc : Integer
                Number of mfccs to produce
            mean : Boolean
                Whether to mean the MFCCs by each rows
            remix : Boolean
                Whether to remix the audio before the extraction of the MFCCs
            align_zero : Boolean
                If True, interval boundaries are mapped to the closest zero-crossing in y
            max_ms : int
                For padding of truncation of audio
        Returns
        ----------
            mfcc : Numpy array 
                Numpy array of the resultant MFCCs

        """
        y_trim = self.ynumpy
        if remix:
            y_trim = librosa.effects.remix(self.ynumpy, intervals=librosa.effects.split(self.ynumpy), align_zeros=align_zero)
            y_trim = self.pad_trunc_function(y_trim, self.sr, max_ms, 1)

        if mean:
            mfcc = np.mean(librosa.feature.mfcc(y=y_trim, sr=self.sr, n_mfcc= n_mfcc).T, axis = 0)
        else:
            mfcc = librosa.feature.mfcc(y=y_trim, sr=self.sr, n_mfcc= n_mfcc).T
        return mfcc

    def MFCC_delta(self, order = 1):
        """
        Produces MFCC delta from the wav file in the dataloader

        Parameters
        ----------
            order : Integer
                the order of the difference operator. 1 for first derivative, 2 for second, etc.
        Returns
        ----------
            delta : Numpy array 
                Numpy array of the resultant MFCC Delta

        """
        delta = librosa.feature.delta(self.mfcc, order = order)
        return delta

    def spectro_augment(self, max_mask_pct=0.1,n_freq_masks=1,n_time_masks=1, condensed = False):
        """
        Produces spectrogram with time masking as well as frequency masking, masking done randomly

        Parameters
        ----------
            max_mask_pct : Float
                Max percentage to mask from each 
            n_freq_masks : Integar
                Number of freq mask (Horizontal)
            n_time_masks : Integar
                Number of time mask (Vertical)
            condensed : Boolean
                Mean of the resultant augmented Spectrogram
        Returns
        ----------
            aug_spec : Numpy array 
                Numpy array of the resultant Mel Spectrogram with the masking
        """
        spec = self.spectro_gram()
        novalue, n_mels, n_steps = np.array(spec).shape
        mask_value = spec.mean()
        aug_spec = torch.from_numpy(spec)

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        if condensed:
            aug_spec = np.array(aug_spec[0]).mean(axis=0)

        return np.array(aug_spec)
    

    def chroma_extraction(self):
        """
        Produces the chroma from the wav file

        Returns
        ----------
            chroma : Numpy array 
                Numpy array of the resultant chroma_stft, which is meaned
        """
        stft = np.abs(librosa.stft(self.ynumpy))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=self.sr).T,axis=0)
        return chroma

    def spectralContract_extraction(self):
        """
        Produces the spectral contract from the wav file

        Returns
        ----------
            contrast : Numpy array 
                Numpy array of the resultant spectral contract, which is meaned 
        """
        stft = np.abs(librosa.stft(self.ynumpy))
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=self.sr).T,axis=0)
        return contrast
    
    def tonalCentroid_extraction(self):
        """
        Produces the tonal centroid from the wav file

        Returns
        ----------
            tonnetz : Numpy array 
                Numpy array of the resultant tonal centroid, which is meaned 
        """
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(self.ynumpy),sr=self.sr).T,axis=0)
        return tonnetz

    def extract_all_features(self, is_concat = False, spectro_augment = False):
        """
        Produces the tonal centroid from the wav file

        Parameters
        ----------
            is_concat : Boolean
                Concate all the features together 
            spectro_augment : Boolean
                Whether masking is done on the Mel_Spectrogram
        Returns
        ----------
            mfcc : Numpy array 
                Numpy array of the resultant MFCC with the masking
            chroma : Numpy array 
                Numpy array of the resultant chroma_stft, which is meaned 
            mel : Numpy array 
                Numpy array of the resultant Mel Spectrogram with the masking
            contrast : Numpy array 
                Numpy array of the resultant spectral contract, which is meaned 
            tonnetz : Numpy array 
                Numpy array of the resultant tonal centroid, which is meaned 
        """
        mfcc = self.MFCC_extraction()
        chroma = self.chroma_extraction()
        if (spectro_augment):
            mel = self.spectro_augment(condensed=True)
        else:
            mel = self.spectro_gram(condensed=True)
        contrast = self.spectralContract_extraction()
        tonnetz = self.tonalCentroid_extraction()

        #mel = self.spectro_gram()
        #mel = np.array(self.spectro_augment(mel, n_freq_masks=2, n_time_masks=2))

        if is_concat == True:
            return np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
            
        return mfcc, chroma, mel, contrast, tonnetz

    def plot_spectrogram(self, augment = False):
        """
        Plot the spectrogram out

        Parameters
        ----------
            augment : Boolean
                Mask the spectrogram 
        """
        if (augment):
            librosa.display.specshow(self.spectro_augment()[0], sr=self.sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
        else:
            librosa.display.specshow(self.spectro_gram()[0], sr=self.sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            
    def pad_trunc_function(self, sig, sr, max_ms, num_rows = 1):
        sig_len = sig.shape[0]
        max_len = sr//1000 * max_ms
        if (sig_len > max_len):
            sig = sig[:,:max_len]
        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = np.zeros((num_rows, pad_begin_len))
            pad_end = np.zeros((num_rows, pad_end_len))
            sig = np.concatenate((pad_begin[0], sig, pad_end[0]), 0)

        return sig

    def melScaled_extraction(self):
        mel = np.mean(librosa.feature.melspectrogram(y = self.ynumpy, sr=self.sr, n_fft = 1024, n_mels= 64, hop_length=None), axis = 0) #np.mean(...(,axis = 0))
        return mel
    
