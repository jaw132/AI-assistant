# import libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write
import bots.TriggerDetection.trigger_utils as tu

Tx, Ty = 5511, 1375
fs = 44100  # Sample rate
seconds = 10  # Duration of recording

def detect_triggerword(input, model):
    # the spectrogram input is (freqs, Tx) and we want (Tx, freqs) to input into the model
    # input = torch.from_numpy(input)
    input = torch.reshape(input, (1, 101, Tx))

    predictions = model(input.detach().float()).detach().numpy()


    for i in range(Ty):
        if predictions[0, i, 0] > 0.5:
            print("hello")
    return predictions

def show_predictions(predictions):
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()

def main():
    model = torch.load("trigger_word_model")

    # records 10 second clip of user and runs through model
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('myrecording.wav', fs, myrecording)

    my_spec = tu.graph_spectrogram("myrecording.wav")
    wake_preds = detect_triggerword(model, my_spec)
    show_predictions(wake_preds)


if __name__ == "__main__":
    main()