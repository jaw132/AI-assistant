import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pydub import AudioSegment

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio():
    activates = []
    backgrounds = []
    negatives = []

    add_wav_to_list("data/activates/", activates)
    add_wav_to_list("data/backgrounds/", backgrounds)
    add_wav_to_list("data/negatives/", negatives)

    return activates, negatives, backgrounds


def add_wav_to_list(path, audio_list):
    for filename in os.listdir(path):
        if filename.endswith("wav"):
            audio_file = AudioSegment.from_wav(path+filename)
            audio_list.append(audio_file)
