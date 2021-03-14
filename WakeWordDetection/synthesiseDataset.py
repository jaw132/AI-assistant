'''The idea is to create a synthesised dataset by overlaying background noise
with the wake word we want to detect and a random assortment of other words'''

# import libraries
import numpy as np
import random
import bots.TriggerDetection.trigger_utils as tu

#define constants
Tx = 5511
n_freq = 101
Ty = 1375
activates, negatives, backgrounds = tu.load_raw_audio()

# Select random time segment of 10sec clip
def get_random_time_segment(segment_ms):
    # Make sure segment doesn't run past the 10sec background
    segment_start = np.random.randint(low=0, high=10000 - segment_ms)
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)

# check if new time segment overlaps with the previous segments
def is_overlapping(segment_time, previous_segments):
    start, end = segment_time[0], segment_time[1]
    overlap = False
    for seg in previous_segments:
        if start <= seg[1] and seg[0] <= end:
            overlap = True

    return overlap

# add a new audio clip to the background audio
def insert_audio_clip(background, audio_clip, previous_segments):
    # add clause if this takes longer than 50 attempts then return a certain number
    # and so not adding overlay and tells calling function not to add labels
    seg_length = len(audio_clip)
    count = 0
    while 1:
        count += 1
        seg_time = get_random_time_segment(seg_length)
        if not is_overlapping(seg_time, previous_segments):
            break
        if count > 50:
            return background, seg_time, 1
    previous_segments.append(seg_time)

    # add segments to background
    new_background = background.overlay(audio_clip, position = seg_time[0])

    return new_background, seg_time, 0

# when we add a wake word to the audio clip we want the labels to be one from
# the end of the wake woke to 50 increments ahead of the end.
def insert_ones(y, segment_end_ms):
  Ty = len(y[0])

  ones_start = int(segment_end_ms*(Ty/10000))+1
  ones_end = min(ones_start+50, Ty-1)

  one_vec = [1]*(ones_end-ones_start)

  y[0, ones_start:ones_end] = one_vec

  return y


def create_training_example(background, activates, negatives):
    y = np.zeros((1, Ty))
    previous_segments = []
    # if both number of activates and negatives are zero then redo until one is >0
    number_activates, number_negatives = 0, 0
    while number_activates == 0 and number_negatives == 0:
        number_activates, number_negatives = np.random.randint(low=1, high=5), np.random.randint(low=0, high=3)

    print(number_activates, number_negatives)

    for act in range(number_activates):
        # pick random activate
        activate = random.choice(activates)
        background, seg_time, ret_code = insert_audio_clip(background, activate, previous_segments)
        if ret_code == 0:
            seg_start, seg_end = seg_time
            y = insert_ones(y, seg_end)

    for neg in range(number_negatives):
        negative = random.choice(negatives)
        background, seg_time, ret_code = insert_audio_clip(background, negative, previous_segments)

    background = tu.match_target_amplitude(background, -20.0)

    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")

    x = tu.graph_spectrogram("train.wav")

    return x, y

# build more examples and store in train set
train_samples, test_samples = 300, 5

# examples are produced as numpy arrays so stick a load together
def create_dataset(train_samps, test_samps):
  X_train, Y_train = [], []
  X_test, Y_test = [], []

  for i in range(train_samps):
    back = random.choice(backgrounds)
    x, y = create_training_example(back, activates, negatives)
    X_train.append(x)
    Y_train.append(y)

  X_train = np.array(X_train)
  Y_train = np.array(Y_train)
  X_train_tensor = torch.from_numpy(X_train)
  Y_train_tensor = torch.from_numpy(Y_train)

  for i in range(test_samps):
    back = random.choice(backgrounds)
    x, y = create_training_example(back, activates, negatives)
    X_test.append(x)
    Y_test.append(y)

  X_test = np.array(X_test)
  Y_test = np.array(Y_test)
  X_test_tensor = torch.from_numpy(X_test)
  Y_test_tensor = torch.from_numpy(Y_test)

  return X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor


def main():
    X_tensor, Y_tensor, X_test_tensor, Y_test_tensor=create_dataset(train_samples, test_samples)

    #save X, Y in a npy file, load into notebook train model there and bring back
    torch.save(X_tensor, "data/train_audio")
    torch.save(Y_tensor, "data/train_labels")
    torch.save(X_test_tensor, "data/test_audio")
    torch.save(Y_test_tensor, "data/test_labels")

if __name__ == "__main__":
    main()
