import numpy as np
import os
from sklearn.utils import shuffle


n_neurons = 100
n_steps = 60
learning_rate = 0.005
n_epochs = 20
BATCH_SIZE = n_steps
SESSION_NUM = "0"
min_time_series_length = 10
max_time_series_length = 20
feature_columns = ['open', 'high', 'low', 'close', 'volume']
n_inputs = len(feature_columns)
n_oututs = n_inputs
SRC_FOLDER_PATH = os.path.dirname(os.getcwd())
CHECKPOINT_PATH = os.path.join(SRC_FOLDER_PATH, 'model_checkpoint', SESSION_NUM)
CHECKPOINT_FOLDER_PATH = os.path.join(CHECKPOINT_PATH, 'SP_model')


def get_next_batch(data, batch_num):
    batch_data = data[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]
    return batch_data


def extract_batches_and_labels(data, sequence_lengths):
    batch_start = 0
    features = []
    labels = []
    for sequence_length in sequence_lengths:
        zeros_array = np.zeros([max_time_series_length - sequence_length, len(feature_columns)], dtype=float)
        print(zeros_array)
        labels.append(np.concatenate((data[batch_start: batch_start + sequence_length], zeros_array)))
        print(labels)
        features.append(np.concatenate((data[batch_start: batch_start + sequence_length], zeros_array)))
        if sequence_length == 10:
            batch_start = + 1
    features_np = np.asarray(features)
    labels_np = np.asarray(labels)
    return features_np, labels_np


def split_test_train_sets(features, labels, sequence_lengths):
    len_training_set = int(len(sequence_lengths) * 0.66)
    sequence_lengths_np = np.asarray(sequence_lengths)
    features_shuffled, labels_shuffled, sequence_lengths_shuffled = shuffle(features, labels, sequence_lengths_np)
    features_train, features_test = np.split(features_shuffled, [len_training_set])
    labels_train, labels_test = np.split(labels_shuffled, [len_training_set])
    sequence_length_train, sequence_length_test = np.split(sequence_lengths_shuffled, [len_training_set])
    return features_train, features_test, labels_train, labels_test, sequence_length_train, sequence_length_test
