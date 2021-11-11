# import libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write
import bots.TriggerDetection.trigger_utils as tu
import time
from pydub import AudioSegment
import bots.TriggerDetection.trigger_utils as tu
from multiprocessing import Process

Tx, Ty = 5511, 1375
fs = 44100  # Sample rate
seconds = 10  # Duration of recording

def detect_triggerword(input, model):
    # the spectrogram input is (freqs, Tx) and we want (Tx, freqs) to input into the model
    # input = torch.from_numpy(input)
    input = torch.reshape(input, (1, 101, Tx))

    predictions = model(input.detach().float()).detach().numpy()

    return predictions

def show_predictions(predictions):
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()

def listen():
    listen_flag, count = 1, 0
    while listen_flag:
        fs = 44100  # Sample rate
        seconds = 2  # Duration of recording
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        write('recording.wav', fs, myrecording)
        count +=1
        if count == 10:
            listen_flag = False


def check_for_activate(filename, model):
    start = time.time()
    background = AudioSegment.from_wav("data/backgrounds/1.wav")
    while 1:
        audio_clip = AudioSegment.from_wav(filename)
        new_background = background.overlay(audio_clip, position=4000)
        new_background.export("user_input" + ".wav", format="wav")

        user_spec = tu.graph_spectrogram("user_input.wav")
        user_preds = detect_triggerword(model, user_spec)

        for i in range(Ty):
            if user_preds[0, i, 0] > 0.5:
                print("hello")
                return 1
        end = time.time()

        if end - start > 20:
            return 0

def main():
    model = torch.load("trigger_word_model")

    p1 = Process(target=listen)
    p1.start()
    p2 = Process(target=check_for_activate("user_input.wav", model))
    p2.start()

    if p2 == 1:
        print("assistant activated")



if __name__ == "__main__":
    main()